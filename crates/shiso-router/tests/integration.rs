//! Integration tests for `shiso-router`.

#![expect(
    clippy::unwrap_used,
    clippy::pedantic,
    reason = "integration tests use unwrap on test setup and wiremock assertions"
)]

use std::net::SocketAddr;
use std::path::PathBuf;

use wiremock::matchers::{body_json_string, header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use shiso_router::config::Config;
use shiso_router::state::AppState;
use shiso_router::tokens::Db;

struct TestRouter {
    addr: SocketAddr,
    admin_token: String,
    client: reqwest::Client,
    sqlite_path: PathBuf,
    _tmp: tempfile::TempDir,
}
#[derive(serde::Deserialize, Default)]
struct TestSeed {
    #[serde(default)]
    exclusive_groups: Vec<TestGroupSeed>,
    #[serde(default)]
    backends: Vec<TestBackendSeed>,
    #[serde(default)]
    models: Vec<TestModelSeed>,
}

#[derive(serde::Deserialize)]
struct TestGroupSeed {
    id: String,
    min_hold_ms: i64,
    max_hold_ms: i64,
    drain_timeout_ms: i64,
}

#[derive(serde::Deserialize)]
struct TestBackendSeed {
    name: String,
    #[serde(deserialize_with = "test_kind_one_or_many")]
    kind: Vec<shiso_router::catalog::BackendKind>,
    base_url: String,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    anthropic_version: Option<String>,
    #[serde(default = "default_true")]
    inject_usage: bool,
    #[serde(default)]
    scheduling_class: shiso_router::catalog::SchedulingClass,
    #[serde(default)]
    exclusive_group: Option<String>,
    #[serde(default)]
    max_concurrency: Option<u32>,
    #[serde(default)]
    max_tokens_cap: Option<i64>,
    #[serde(default)]
    tok_s_estimate: Option<f64>,
}

fn default_true() -> bool {
    true
}

fn test_kind_one_or_many<'de, D>(de: D) -> Result<Vec<shiso_router::catalog::BackendKind>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(shiso_router::catalog::BackendKind),
        Many(Vec<shiso_router::catalog::BackendKind>),
    }
    Ok(match <OneOrMany as serde::Deserialize>::deserialize(de)? {
        OneOrMany::One(k) => vec![k],
        OneOrMany::Many(v) => v,
    })
}

#[derive(serde::Deserialize)]
struct TestModelSeed {
    name: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    pricing: Option<shiso_router::catalog::PricingCfg>,
    #[serde(default)]
    qos: Option<TestQosSeed>,
    #[serde(default)]
    on_shed_without_fallback: shiso_router::catalog::OnShedWithoutFallback,
    #[serde(default)]
    targets: Vec<TestTargetSeed>,
}

#[derive(serde::Deserialize)]
struct TestQosSeed {
    max_wait_ms: Option<i64>,
    background_max_wait_ms: Option<i64>,
}

#[derive(serde::Deserialize)]
struct TestTargetSeed {
    backend: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default = "default_weight")]
    weight: u32,
    #[serde(default)]
    pricing: Option<shiso_router::catalog::PricingCfg>,
    #[serde(default)]
    fallback_only: bool,
}

fn default_weight() -> u32 {
    1
}

fn seed_test_db(db: &Db, config_body: &str) {
    if config_body.trim().is_empty() {
        return;
    }
    let seed: TestSeed = toml::from_str(config_body).expect("valid test seed toml");
    for g in seed.exclusive_groups {
        let def = shiso_router::catalog::GroupDef {
            id: g.id,
            min_hold_ms: g.min_hold_ms,
            max_hold_ms: g.max_hold_ms,
            drain_timeout_ms: g.drain_timeout_ms,
        };
        shiso_router::catalog::insert_group(db, &def).expect("insert test group");
    }
    for b in seed.backends {
        let def = shiso_router::catalog::BackendDef {
            name: b.name,
            kinds: b.kind,
            base_url: b.base_url,
            api_key: b.api_key,
            api_key_env: b.api_key_env,
            anthropic_version: b.anthropic_version,
            inject_usage: b.inject_usage,
            scheduling_class: b.scheduling_class,
            exclusive_group_id: b.exclusive_group,
            max_concurrency: b.max_concurrency,
            max_tokens_cap: b.max_tokens_cap,
            tok_s_estimate: b.tok_s_estimate,
        };
        shiso_router::catalog::insert_backend(db, &def).expect("insert test backend");
    }
    for m in seed.models {
        let def = shiso_router::catalog::ModelDef {
            name: m.name,
            aliases: m.aliases,
            pricing: m.pricing,
            qos_max_wait_ms: m.qos.as_ref().and_then(|q| q.max_wait_ms),
            qos_background_max_wait_ms: m.qos.as_ref().and_then(|q| q.background_max_wait_ms),
            on_shed_without_fallback: m.on_shed_without_fallback,
            targets: m
                .targets
                .into_iter()
                .map(|t| shiso_router::catalog::TargetDef {
                    backend: t.backend,
                    model: t.model,
                    weight: t.weight,
                    pricing: t.pricing,
                    fallback_only: t.fallback_only,
                })
                .collect(),
        };
        shiso_router::catalog::insert_model(db, &def).expect("insert test model");
    }
}

impl TestRouter {
    /// Boots the router in-process on an ephemeral port with a temp SQLite
    /// database and the given config body (listen/sqlite_path prepended).
    async fn start(config_body: &str) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let sqlite_path: PathBuf = tmp.path().join("tokens.db");
        let admin_token = "test-admin-token".to_string();

        let raw = format!(
            "listen = \"127.0.0.1:0\"\nsqlite_path = {:?}\nadmin_token = {:?}\n",
            sqlite_path.to_str().unwrap(),
            admin_token,
        );
        let cfg: Config = toml::from_str(&raw).unwrap();
        cfg.validate().unwrap();

        let db = Db::open(&cfg.sqlite_path).unwrap();
        seed_test_db(&db, config_body);
        let (log_sink, _log_writer) = shiso_router::logging::start_writer(db.clone());
        let state = AppState::from_config(&cfg, db, log_sink).unwrap();
        let app = shiso_router::build_router(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        TestRouter {
            addr,
            admin_token,
            client: reqwest::Client::new(),
            sqlite_path,
            _tmp: tmp,
        }
    }

    /// Like `start`, but with `admin_listen` configured: the admin surface
    /// is served on a second ephemeral listener whose address is returned.
    async fn start_with_admin_listener(config_body: &str) -> (Self, SocketAddr) {
        let tmp = tempfile::tempdir().unwrap();
        let sqlite_path: PathBuf = tmp.path().join("tokens.db");
        let admin_token = "test-admin-token".to_string();

        // The configured socket addrs are placeholders — both listeners
        // are bound on ephemeral ports below; admin_listen only needs to
        // be present (and distinct) so admin_on_main becomes false.
        let raw = format!(
            "listen = \"127.0.0.1:0\"\nadmin_listen = \"127.0.0.1:9\"\nsqlite_path = {:?}\nadmin_token = {:?}\n",
            sqlite_path.to_str().unwrap(),
            admin_token,
        );
        let cfg: Config = toml::from_str(&raw).unwrap();
        cfg.validate().unwrap();

        let db = Db::open(&cfg.sqlite_path).unwrap();
        seed_test_db(&db, config_body);
        let (log_sink, _log_writer) = shiso_router::logging::start_writer(db.clone());
        let state = AppState::from_config(&cfg, db, log_sink).unwrap();
        let app = shiso_router::build_router(state.clone());
        let admin_app = shiso_router::build_admin_router(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let admin_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let admin_addr = admin_listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(admin_listener, admin_app).await.unwrap();
        });

        (
            TestRouter {
                addr,
                admin_token,
                client: reqwest::Client::new(),
                sqlite_path,
                _tmp: tmp,
            },
            admin_addr,
        )
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}{}", self.addr, path)
    }

    /// Polls the request_logs table (log writes are async) until at least
    /// `expected` rows exist, returning them newest-last as JSON-ish maps.
    async fn wait_for_logs(&self, expected: usize) -> Vec<serde_json::Value> {
        for _ in 0..200 {
            let rows = self.read_logs();
            if rows.len() >= expected {
                return rows;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        panic!("expected {expected} log rows, got {:?}", self.read_logs());
    }

    fn read_logs(&self) -> Vec<serde_json::Value> {
        let conn = rusqlite::Connection::open_with_flags(
            &self.sqlite_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, token_id, model, backend, upstream_model, path, status,
                        streamed, ttfb_ms, duration_ms, prompt_tokens, completion_tokens,
                        cached_tokens, cache_write_tokens, total_tokens, cost,
                        finish_reason, error
                 FROM request_logs ORDER BY created_at, id",
            )
            .unwrap();
        stmt.query_map([], |r| {
            Ok(serde_json::json!({
                "id": r.get::<_, String>(0)?,
                "token_id": r.get::<_, String>(1)?,
                "model": r.get::<_, String>(2)?,
                "backend": r.get::<_, Option<String>>(3)?,
                "upstream_model": r.get::<_, Option<String>>(4)?,
                "path": r.get::<_, String>(5)?,
                "status": r.get::<_, Option<i64>>(6)?,
                "streamed": r.get::<_, bool>(7)?,
                "ttfb_ms": r.get::<_, Option<i64>>(8)?,
                "duration_ms": r.get::<_, Option<i64>>(9)?,
                "prompt_tokens": r.get::<_, Option<i64>>(10)?,
                "completion_tokens": r.get::<_, Option<i64>>(11)?,
                "cached_tokens": r.get::<_, Option<i64>>(12)?,
                "cache_write_tokens": r.get::<_, Option<i64>>(13)?,
                "total_tokens": r.get::<_, Option<i64>>(14)?,
                "cost": r.get::<_, Option<f64>>(15)?,
                "finish_reason": r.get::<_, Option<String>>(16)?,
                "error": r.get::<_, Option<String>>(17)?,
            }))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
    }

    async fn create_token(&self) -> (String, String) {
        let resp = self
            .client
            .post(self.url("/admin/tokens"))
            .bearer_auth(&self.admin_token)
            .json(&serde_json::json!({"name": "test"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 201);
        let body: serde_json::Value = resp.json().await.unwrap();
        (
            body["token"].as_str().unwrap().to_string(),
            body["id"].as_str().unwrap().to_string(),
        )
    }
}

fn backend_toml(name: &str, base_url: &str, key: &str) -> String {
    format!(
        "[[backends]]\nname = {name:?}\nkind = \"openai\"\nbase_url = {base_url:?}\napi_key = {key:?}\n"
    )
}

const CHAT_BODY: &str = r#"{"model":"gpt-test","messages":[{"role":"user","content":"hi"}]}"#;

#[tokio::test]
async fn token_lifecycle_and_auth() {
    let router = TestRouter::start(
        &(backend_toml("b1", "http://127.0.0.1:1", "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;

    // Admin endpoints reject bad admin tokens.
    let resp = router
        .client
        .get(router.url("/admin/tokens"))
        .bearer_auth("wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    // Create, list, delete.
    let (token, id) = router.create_token().await;
    assert!(token.starts_with("sk-"));

    let list: serde_json::Value = router
        .client
        .get(router.url("/admin/tokens"))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list.as_array().unwrap().len(), 1);
    assert_eq!(list[0]["name"], "test");
    assert!(
        list[0].get("token").is_none(),
        "list must not expose secrets"
    );
    assert!(list[0].get("token_hash").is_none());

    // Proxy rejects missing/invalid tokens.
    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth("sk-invalid")
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    // Revoked token stops working immediately.
    let resp = router
        .client
        .delete(router.url(&format!("/admin/tokens/{id}")))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 204);

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    // Deleting again -> 404.
    let resp = router
        .client
        .delete(router.url(&format!("/admin/tokens/{id}")))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn forwards_with_backend_key_and_streams_sse() {
    let mock = MockServer::start().await;
    let sse_body = "data: {\"chunk\":1}\n\ndata: {\"chunk\":2}\n\ndata: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer backend-secret"))
        // Body must be forwarded byte-for-byte (no rename configured).
        .and(body_json_string(CHAT_BODY))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse_body, "text/event-stream"),
        )
        .expect(1)
        .mount(&mock)
        .await;

    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "backend-secret")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .header("content-type", "application/json")
        .header("user-agent", "OpenAI/Go 3.41.0")
        .header("x-stainless-lang", "go")
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers()["content-type"], "text/event-stream");
    assert_eq!(resp.text().await.unwrap(), sse_body);

    // The client's router token must not have leaked upstream.
    let received = mock.received_requests().await.unwrap();
    assert_eq!(
        received[0].headers.get("authorization").unwrap(),
        "Bearer backend-secret"
    );
    // ...but its User-Agent must reach the backend: providers vary their
    // stream framing by SDK user agent.
    assert_eq!(
        received[0].headers.get("user-agent").unwrap(),
        "OpenAI/Go 3.41.0"
    );
    assert_eq!(received[0].headers.get("x-stainless-lang").unwrap(), "go");
}

#[tokio::test]
async fn renames_model_when_configured() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "anthropic-secret"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
        .expect(1)
        .mount(&mock)
        .await;

    let config = format!(
        "[[backends]]\nname = \"claude\"\nkind = \"anthropic\"\nbase_url = {:?}\napi_key = \"anthropic-secret\"\n\
         [[models]]\nname = \"claude-alias\"\n[[models.targets]]\nbackend = \"claude\"\nmodel = \"claude-real-model\"\n",
        mock.uri()
    );
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/messages"))
        .header("x-api-key", &token)
        .body(r#"{"model":"claude-alias","max_tokens":10,"messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let received: Vec<Request> = mock.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["model"], "claude-real-model", "model must be renamed");
    assert_eq!(body["max_tokens"], 10, "other fields preserved");
}

#[tokio::test]
async fn mixed_kind_route_serves_each_endpoint_from_matching_backend() {
    let openai_mock = MockServer::start().await;
    let anthropic_mock = MockServer::start().await;

    // Each backend must only ever see requests in the wire format it
    // speaks — 4 requests per endpoint, all landing on the matching mock.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": "openai"})))
        .expect(4)
        .mount(&openai_mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": "anthropic"})),
        )
        .expect(4)
        .mount(&anthropic_mock)
        .await;

    let config = format!(
        "[[backends]]\nname = \"oai\"\nkind = \"openai\"\nbase_url = {:?}\napi_key = \"k1\"\n\
         [[backends]]\nname = \"ant\"\nkind = \"anthropic\"\nbase_url = {:?}\napi_key = \"k2\"\n\
         [[models]]\nname = \"dual\"\n\
         [[models.targets]]\nbackend = \"oai\"\n\
         [[models.targets]]\nbackend = \"ant\"\n",
        openai_mock.uri(),
        anthropic_mock.uri()
    );
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    for _ in 0..4 {
        let resp = router
            .client
            .post(router.url("/v1/chat/completions"))
            .bearer_auth(&token)
            .body(r#"{"model":"dual","messages":[]}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }
    for _ in 0..4 {
        let resp = router
            .client
            .post(router.url("/v1/messages"))
            .bearer_auth(&token)
            .body(r#"{"model":"dual","max_tokens":10,"messages":[]}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }
}

#[tokio::test]
async fn dual_kind_backend_gets_auth_per_endpoint() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer gw-secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": 1})))
        .expect(1)
        .mount(&mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "gw-secret"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": 2})))
        .expect(1)
        .mount(&mock)
        .await;

    let config = format!(
        "[[backends]]\nname = \"gw\"\nkind = [\"openai\", \"anthropic\"]\nbase_url = {:?}\napi_key = \"gw-secret\"\n\
         [[models]]\nname = \"dual\"\n[[models.targets]]\nbackend = \"gw\"\n",
        mock.uri()
    );
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(r#"{"model":"dual","messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = router
        .client
        .post(router.url("/v1/messages"))
        .bearer_auth(&token)
        .body(r#"{"model":"dual","max_tokens":10,"messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn prefixless_aliases_and_v1_suffixed_base_url() {
    let mock = MockServer::start().await;
    // The backend URL already ends in /v1, so the upstream path must be
    // /v1/responses — never /v1/v1/responses.
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": 1})))
        .expect(2)
        .mount(&mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": 2})))
        .expect(1)
        .mount(&mock)
        .await;

    let config = format!(
        "[[backends]]\nname = \"gw\"\nkind = [\"openai\", \"anthropic\"]\nbase_url = {:?}\napi_key = \"gw-secret\"\n\
         [[models]]\nname = \"dual\"\n[[models.targets]]\nbackend = \"gw\"\n",
        format!("{}/v1", mock.uri())
    );
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    for path in ["/responses", "/v1/responses"] {
        let resp = router
            .client
            .post(router.url(path))
            .bearer_auth(&token)
            .body(r#"{"model":"dual","input":"hi"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "POST {path}");
    }

    let resp = router
        .client
        .post(router.url("/messages"))
        .bearer_auth(&token)
        .body(r#"{"model":"dual","max_tokens":10,"messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // The alias is canonicalized before logging, so both spellings of an
    // endpoint appear as one path.
    let logs = router.wait_for_logs(3).await;
    let mut paths: Vec<&str> = logs.iter().map(|r| r["path"].as_str().unwrap()).collect();
    paths.sort();
    assert_eq!(paths, ["/v1/messages", "/v1/responses", "/v1/responses"]);
}

/// The scoring endpoints ride the proxy like any other JSON endpoint:
/// both path spellings reach the upstream as the canonical path, and the
/// usage row parses in each wire's spelling (`prompt_tokens` on decisions,
/// `input_tokens` on systemone).
#[tokio::test]
async fn scoring_endpoints_are_proxied_and_usage_logged() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/decisions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "decisions", "model": "gpt-test", "prompt_format_version": 1,
            "answers": {"q1": {"type": "yes_no", "answer": "no", "probabilities": [0.02, 0.98], "label_mass": 1.0}},
            "usage": {"prompt_tokens": 42, "completion_tokens": 0, "total_tokens": 42}
        })))
        .expect(2)
        .mount(&mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "model": "gpt-test", "state": "",
            "answers": {"q1": {"type": "yes_no", "answer": "no", "confidence": 0.96, "x_label_mass": 1.0}},
            "usage": {"input_tokens": 17, "output_tokens": 0, "total_tokens": 17}
        })))
        .expect(2)
        .mount(&mock)
        .await;

    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    for path in ["/v1/decisions", "/decisions", "/v1/systemone", "/systemone"] {
        let resp = router
            .client
            .post(router.url(path))
            .bearer_auth(&token)
            .body(r#"{"model":"gpt-test","input":"the ticket","questions":[{"id":"q1","type":"yes_no","question":"Is this about billing?"}]}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "POST {path}");
    }

    let mut logs = router.wait_for_logs(4).await;
    logs.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
    let paths: Vec<&str> = logs.iter().map(|r| r["path"].as_str().unwrap()).collect();
    assert_eq!(
        paths,
        [
            "/v1/decisions",
            "/v1/decisions",
            "/v1/systemone",
            "/v1/systemone"
        ]
    );
    // Both usage spellings normalize to the same row shape: decisions
    // rows carry 42 (prompt_tokens), systemone rows 17 (input_tokens).
    let prompts: Vec<i64> = logs
        .iter()
        .map(|r| r["prompt_tokens"].as_i64().unwrap())
        .collect();
    assert_eq!(prompts, [42, 42, 17, 17], "rows: {logs:?}");
    for row in &logs {
        assert_eq!(row["completion_tokens"].as_i64(), Some(0));
    }
}

#[tokio::test]
async fn trace_header_is_not_forwarded_upstream() {
    let mock = MockServer::start().await;
    let sse_body = "data: {\"chunk\":1}\n\ndata: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse_body, "text/event-stream"),
        )
        .expect(1)
        .mount(&mock)
        .await;

    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "backend-secret")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .header("content-type", "application/json")
        .header("x-debug", "trace")
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // Tracing must not alter the relayed bytes.
    assert_eq!(resp.text().await.unwrap(), sse_body);

    let received = mock.received_requests().await.unwrap();
    assert!(
        received[0].headers.get("x-debug").is_none(),
        "x-debug must stay router-local"
    );

    // The traced request is logged like any other, under the trace id.
    let logs = router.wait_for_logs(1).await;
    assert_eq!(logs[0]["status"], 200);
}

#[tokio::test]
async fn rejects_endpoint_without_matching_backend() {
    // The route exists but only has an anthropic target, so an
    // OpenAI-format endpoint has nothing to serve it.
    let config = "[[backends]]\nname = \"ant\"\nkind = \"anthropic\"\nbase_url = \"http://127.0.0.1:1\"\napi_key = \"k\"\n\
         [[models]]\nname = \"claude-only\"\n[[models.targets]]\nbackend = \"ant\"\n";
    let router = TestRouter::start(config).await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(r#"{"model":"claude-only","messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "not_found_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no backend serving /v1/chat/completions")
    );

    let logs = router.wait_for_logs(1).await;
    assert_eq!(logs[0]["status"], 404);
    assert!(
        logs[0]["error"]
            .as_str()
            .unwrap()
            .contains("no backend serving")
    );
}

#[tokio::test]
async fn fails_over_to_secondary_on_5xx() {
    let primary = MockServer::start().await;
    let secondary = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .expect(1)
        .mount(&primary)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"from": "secondary"})),
        )
        .expect(1)
        .mount(&secondary)
        .await;

    // weight 1000000 vs 1: the first pick is deterministic (primary).
    let config = backend_toml("primary", &primary.uri(), "k1")
        + &backend_toml("secondary", &secondary.uri(), "k2")
        + "[[models]]\nname = \"gpt-test\"\n\
           [[models.targets]]\nbackend = \"primary\"\nweight = 1000000\n\
           [[models.targets]]\nbackend = \"secondary\"\nweight = 1\n";
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["from"], "secondary");
}

#[tokio::test]
async fn relays_last_5xx_when_all_backends_fail() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(503).set_body_json(serde_json::json!({"error": "overloaded"})),
        )
        .mount(&mock)
        .await;

    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503, "upstream 5xx relayed verbatim");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"], "overloaded");
}

#[tokio::test]
async fn returns_502_when_backend_unreachable() {
    // Port 1 on localhost: connection refused.
    let router = TestRouter::start(
        &(backend_toml("dead", "http://127.0.0.1:1", "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"dead\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "api_error");
}

#[tokio::test]
async fn unknown_model_and_bad_body() {
    let router = TestRouter::start(
        &(backend_toml("b1", "http://127.0.0.1:1", "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(r#"{"model":"nope","messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "not_found_error");

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body("not json at all")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn upstream_4xx_is_relayed_without_failover() {
    let primary = MockServer::start().await;
    let secondary = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429).set_body_json(serde_json::json!({"error": "rate limited"})),
        )
        .expect(1)
        .mount(&primary)
        .await;
    // Secondary must never be called for a 4xx.
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&secondary)
        .await;

    let config = backend_toml("primary", &primary.uri(), "k1")
        + &backend_toml("secondary", &secondary.uri(), "k2")
        + "[[models]]\nname = \"gpt-test\"\n\
           [[models.targets]]\nbackend = \"primary\"\nweight = 1000000\n\
           [[models.targets]]\nbackend = \"secondary\"\nweight = 1\n";
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 429);
}

#[tokio::test]
async fn load_is_balanced_across_backends() {
    let a = MockServer::start().await;
    let b = MockServer::start().await;
    for m in [&a, &b] {
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(m)
            .await;
    }

    let config = backend_toml("a", &a.uri(), "ka")
        + &backend_toml("b", &b.uri(), "kb")
        + "[[models]]\nname = \"gpt-test\"\n\
           [[models.targets]]\nbackend = \"a\"\nweight = 3\n\
           [[models.targets]]\nbackend = \"b\"\nweight = 1\n";
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    for _ in 0..8 {
        let resp = router
            .client
            .post(router.url("/v1/responses"))
            .bearer_auth(&token)
            .body(r#"{"model":"gpt-test","input":"hi"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    assert_eq!(a.received_requests().await.unwrap().len(), 6);
    assert_eq!(b.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn embeddings_are_routed_renamed_and_logged() {
    let mock = MockServer::start().await;
    let upstream_body = serde_json::json!({
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, -0.2, 0.3]}],
        "model": "bge-m3",
        "usage": {"prompt_tokens": 8, "total_tokens": 8}
    });
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header("authorization", "Bearer k"))
        .and(body_json_string(
            r#"{"model":"bge-m3","input":"the quick brown fox","encoding_format":"float"}"#,
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body.clone()))
        .mount(&mock)
        .await;

    let config = backend_toml("b1", &mock.uri(), "k")
        + "[[models]]\nname = \"text-embedding-3-small\"\n\
           [models.pricing]\ninput_per_mtok = 2.0\noutput_per_mtok = 0.0\n\
           [[models.targets]]\nbackend = \"b1\"\nmodel = \"bge-m3\"\n";
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/embeddings"))
        .bearer_auth(&token)
        .header("content-type", "application/json")
        .body(
            r#"{"model":"text-embedding-3-small","input":"the quick brown fox","encoding_format":"float"}"#,
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // The response body is relayed untouched, upstream model name included.
    assert_eq!(
        resp.json::<serde_json::Value>().await.unwrap(),
        upstream_body
    );

    let rows = router.wait_for_logs(1).await;
    let row = &rows[0];
    assert_eq!(row["model"], "text-embedding-3-small");
    assert_eq!(row["upstream_model"], "bge-m3");
    assert_eq!(row["path"], "/v1/embeddings");
    assert_eq!(row["status"], 200);
    assert_eq!(row["streamed"], false);
    assert_eq!(row["prompt_tokens"], 8);
    // Embeddings have no output tokens and no finish reason.
    assert_eq!(row["completion_tokens"], serde_json::Value::Null);
    assert_eq!(row["total_tokens"], 8);
    assert_eq!(row["finish_reason"], serde_json::Value::Null);
    assert!((row["cost"].as_f64().unwrap() - 8.0 / 1e6 * 2.0).abs() < 1e-12);
}

/// A batch embedding response is mostly vector data and can be far larger
/// than the whole-body parse cap; usage still has to come out of it.
#[tokio::test]
async fn large_embeddings_response_still_logs_usage() {
    let vectors: Vec<serde_json::Value> = (0..400)
        .map(|i| {
            serde_json::json!({
                "object": "embedding",
                "index": i,
                "embedding": vec![0.012_345_678_9_f64; 1536],
            })
        })
        .collect();
    let body = serde_json::json!({
        "object": "list",
        "data": vectors,
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 120_000, "total_tokens": 120_000}
    })
    .to_string();
    assert!(
        body.len() > 4 * 1024 * 1024,
        "body must exceed the JSON buffer cap"
    );

    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(body.into_bytes(), "application/json"),
        )
        .mount(&mock)
        .await;

    let config = backend_toml("b1", &mock.uri(), "k")
        + "[[models]]\nname = \"text-embedding-3-small\"\n\
           [models.pricing]\ninput_per_mtok = 0.02\noutput_per_mtok = 0.0\n\
           [[models.targets]]\nbackend = \"b1\"\n";
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/embeddings"))
        .bearer_auth(&token)
        .body(r#"{"model":"text-embedding-3-small","input":["a","b"]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.bytes().await.unwrap().len() > 4 * 1024 * 1024);

    let rows = router.wait_for_logs(1).await;
    assert_eq!(rows[0]["prompt_tokens"], 120_000);
    assert_eq!(rows[0]["total_tokens"], 120_000);
    assert!((rows[0]["cost"].as_f64().unwrap() - 120_000.0 / 1e6 * 0.02).abs() < 1e-12);
}

/// OpenRouter serves embeddings at `<host>/api/v1/embeddings`, uses slug
/// model names, and returns extra `usage` fields around the two that
/// matter.
#[tokio::test]
async fn embeddings_via_openrouter_shaped_backend() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/embeddings"))
        .and(header("authorization", "Bearer sk-or-key"))
        .and(body_json_string(
            r#"{"model":"openai/text-embedding-3-small","input":"hi","dimensions":1536,"input_type":"search_query"}"#,
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "embd-123",
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "openai/text-embedding-3-small",
            "usage": {
                "prompt_tokens": 8, "total_tokens": 8, "cost": 0.0001, "is_byok": false,
                "prompt_tokens_details": {"text_tokens": 8},
                "cost_details": {
                    "upstream_inference_prompt_cost": 0.0001,
                    "upstream_inference_completions_cost": 0.0
                }
            }
        })))
        .mount(&mock)
        .await;

    let config = format!(
        "[[backends]]\nname = \"openrouter\"\nkind = \"openai\"\n\
         base_url = \"{}/api\"\napi_key = \"sk-or-key\"\n\
         [[models]]\nname = \"embed-small\"\n\
         [models.pricing]\ninput_per_mtok = 0.02\noutput_per_mtok = 0.0\n\
         [[models.targets]]\nbackend = \"openrouter\"\nmodel = \"openai/text-embedding-3-small\"\n",
        mock.uri()
    );
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/embeddings"))
        .bearer_auth(&token)
        .body(
            r#"{"model":"embed-small","input":"hi","dimensions":1536,"input_type":"search_query"}"#,
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["usage"]["cost"], 0.0001);

    let rows = router.wait_for_logs(1).await;
    assert_eq!(rows[0]["model"], "embed-small");
    assert_eq!(rows[0]["upstream_model"], "openai/text-embedding-3-small");
    assert_eq!(rows[0]["prompt_tokens"], 8);
    assert_eq!(rows[0]["total_tokens"], 8);
    // OpenRouter's prompt_tokens_details is a modality breakdown, not a
    // cached-token subset.
    assert_eq!(rows[0]["cached_tokens"], serde_json::Value::Null);
}

#[tokio::test]
async fn embeddings_reject_anthropic_only_backend() {
    let config = "[[backends]]\nname = \"ant\"\nkind = \"anthropic\"\n\
                  base_url = \"http://127.0.0.1:1\"\napi_key = \"k\"\n\
                  [[models]]\nname = \"emb\"\n[[models.targets]]\nbackend = \"ant\"\n";
    let router = TestRouter::start(config).await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/embeddings"))
        .bearer_auth(&token)
        .body(r#"{"model":"emb","input":"hi"}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no backend serving /v1/embeddings")
    );
}

/// A transcription request as the OpenAI SDK sends it: a multipart form
/// with the audio file and the text fields beside it.
fn transcription_form(model: &str, stream: bool) -> (String, Vec<u8>) {
    let boundary = "----llmroutertest";
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp3\"\r\n\
             Content-Type: audio/mpeg\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(b"ID3\x00\xff\xfe\r\nnot really audio");
    let mut field = |name: &str, value: &str| {
        body.extend_from_slice(
            format!(
                "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}"
            )
            .as_bytes(),
        );
    };
    field("model", model);
    field("language", "en");
    if stream {
        field("stream", "true");
    }
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

#[tokio::test]
async fn transcriptions_are_routed_renamed_and_logged() {
    let mock = MockServer::start().await;
    let upstream_body = serde_json::json!({
        "text": "hello there",
        "usage": {"type": "tokens", "input_tokens": 14, "output_tokens": 45, "total_tokens": 59,
                  "input_token_details": {"text_tokens": 0, "audio_tokens": 14}}
    });
    let (content_type, _) = transcription_form("x", false);
    let (_, expected_body) = transcription_form("gpt-4o-transcribe", false);
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(header("authorization", "Bearer k"))
        // The boundary must survive, or the upstream cannot parse the form.
        .and(header("content-type", content_type.as_str()))
        .and(wiremock::matchers::body_bytes(expected_body))
        .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body.clone()))
        .mount(&mock)
        .await;

    let config = backend_toml("b1", &mock.uri(), "k")
        + "[[models]]\nname = \"transcribe\"\n\
           [models.pricing]\ninput_per_mtok = 6.0\noutput_per_mtok = 10.0\n\
           [[models.targets]]\nbackend = \"b1\"\nmodel = \"gpt-4o-transcribe\"\n";
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    // Both spellings reach the same handler and log under the /v1 path.
    for route in ["/v1/audio/transcriptions", "/audio/transcriptions"] {
        let (content_type, body) = transcription_form("transcribe", false);
        let resp = router
            .client
            .post(router.url(route))
            .bearer_auth(&token)
            .header("content-type", content_type)
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{route}");
        assert_eq!(
            resp.json::<serde_json::Value>().await.unwrap(),
            upstream_body
        );
    }

    let rows = router.wait_for_logs(2).await;
    for row in &rows {
        assert_eq!(row["model"], "transcribe");
        assert_eq!(row["upstream_model"], "gpt-4o-transcribe");
        assert_eq!(row["path"], "/v1/audio/transcriptions");
        assert_eq!(row["status"], 200);
        assert_eq!(row["prompt_tokens"], 14);
        assert_eq!(row["completion_tokens"], 45);
        assert!((row["cost"].as_f64().unwrap() - (14.0 * 6.0 + 45.0 * 10.0) / 1e6).abs() < 1e-12);
    }
}

#[tokio::test]
async fn streamed_transcription_logs_usage_from_the_done_event() {
    let mock = MockServer::start().await;
    let sse = "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hi\"}\n\n\
               data: {\"type\":\"transcript.text.done\",\"text\":\"hi\",\"usage\":{\"type\":\"tokens\",\"input_tokens\":9,\"output_tokens\":3,\"total_tokens\":12}}\n\n";
    // Forwarded untouched: no rename, and nothing to inject into a form.
    let (_, expected_body) = transcription_form("gpt-4o-mini-transcribe", true);
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(wiremock::matchers::body_bytes(expected_body))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
        .mount(&mock)
        .await;

    let config = backend_toml("b1", &mock.uri(), "k")
        + "[[models]]\nname = \"gpt-4o-mini-transcribe\"\n[[models.targets]]\nbackend = \"b1\"\n";
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    let (content_type, body) = transcription_form("gpt-4o-mini-transcribe", true);
    let resp = router
        .client
        .post(router.url("/v1/audio/transcriptions"))
        .bearer_auth(&token)
        .header("content-type", content_type)
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), sse);

    let rows = router.wait_for_logs(1).await;
    assert_eq!(rows[0]["streamed"], true);
    assert_eq!(rows[0]["prompt_tokens"], 9);
    assert_eq!(rows[0]["completion_tokens"], 3);
}

#[tokio::test]
async fn transcription_requires_a_form_body() {
    let config = backend_toml("b1", "http://127.0.0.1:1", "k")
        + "[[models]]\nname = \"whisper-1\"\n[[models.targets]]\nbackend = \"b1\"\n";
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/audio/transcriptions"))
        .bearer_auth(&token)
        .header("content-type", "application/json")
        .body(r#"{"model":"whisper-1"}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("multipart/form-data")
    );
}

#[tokio::test]
async fn key_controls_disable_allowlist_and_patch() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
        .mount(&mock)
        .await;

    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"
            + "[[models]]\nname = \"other-model\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;

    // Create a key restricted to other-model.
    let resp = router
        .client
        .post(router.url("/admin/tokens"))
        .bearer_auth(&router.admin_token)
        .json(&serde_json::json!({"name": "restricted", "allowed_models": ["other-model"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: serde_json::Value = resp.json().await.unwrap();
    let token = body["token"].as_str().unwrap().to_string();
    let id = body["id"].as_str().unwrap().to_string();
    assert_eq!(body["allowed_models"][0], "other-model");
    assert_eq!(body["disabled"], false);

    // A model outside the allowlist is rejected with 403 permission_error.
    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
    let err: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(err["error"]["type"], "permission_error");

    // The allowed model goes through.
    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(r#"{"model":"other-model","messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Unknown allowlist entries are rejected.
    let resp = router
        .client
        .patch(router.url(&format!("/admin/tokens/{id}")))
        .bearer_auth(&router.admin_token)
        .json(&serde_json::json!({"allowed_models": ["not-a-model"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);

    // PATCH: clear the allowlist, disable the key, set a credit limit.
    let resp = router
        .client
        .patch(router.url(&format!("/admin/tokens/{id}")))
        .bearer_auth(&router.admin_token)
        .json(&serde_json::json!({
            "allowed_models": null,
            "disabled": true,
            "credit_limit": 5.5,
            "credit_limit_reset": "monthly"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["disabled"], true);
    assert_eq!(body["credit_limit"], 5.5);
    assert_eq!(body["credit_limit_reset"], "monthly");
    assert!(body["allowed_models"].is_null());

    // Disabled keys are rejected immediately (cache swap, no restart).
    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
    let err: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(err["error"]["type"], "permission_error");

    // Re-enable; with the allowlist cleared every model works again.
    let resp = router
        .client
        .patch(router.url(&format!("/admin/tokens/{id}")))
        .bearer_auth(&router.admin_token)
        .json(&serde_json::json!({"disabled": false}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // GET reflects the stored state; PATCH of a missing id is a 404.
    let resp = router
        .client
        .get(router.url(&format!("/admin/tokens/{id}")))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["name"], "restricted");
    assert_eq!(body["credit_limit"], 5.5);
    assert!(
        body.get("token").is_none(),
        "detail must not expose secrets"
    );

    let resp = router
        .client
        .patch(router.url("/admin/tokens/missing-id"))
        .bearer_auth(&router.admin_token)
        .json(&serde_json::json!({"disabled": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn logs_non_streaming_request_with_usage_and_cost() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500,
                      "prompt_tokens_details": {"cached_tokens": 200}}
        })))
        .mount(&mock)
        .await;

    let config = backend_toml("b1", &mock.uri(), "k")
        + "[[models]]\nname = \"gpt-test\"\n\
           [models.pricing]\ninput_per_mtok = 2.0\noutput_per_mtok = 10.0\ncached_input_per_mtok = 0.5\n\
           [[models.targets]]\nbackend = \"b1\"\n";
    let router = TestRouter::start(&config).await;
    let (token, id) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let rows = router.wait_for_logs(1).await;
    let row = &rows[0];
    assert_eq!(row["token_id"], id);
    assert_eq!(row["model"], "gpt-test");
    assert_eq!(row["backend"], "b1");
    assert_eq!(row["upstream_model"], "gpt-test");
    assert_eq!(row["path"], "/v1/chat/completions");
    assert_eq!(row["status"], 200);
    assert_eq!(row["streamed"], false);
    assert_eq!(row["prompt_tokens"], 1000);
    assert_eq!(row["completion_tokens"], 500);
    assert_eq!(row["cached_tokens"], 200);
    assert_eq!(row["total_tokens"], 1500);
    assert_eq!(row["finish_reason"], "stop");
    // 800 fresh input + 200 cached + 500 output, per-mtok rates 2.0/0.5/10.0.
    let expected = 800.0 / 1e6 * 2.0 + 200.0 / 1e6 * 0.5 + 500.0 / 1e6 * 10.0;
    assert!((row["cost"].as_f64().unwrap() - expected).abs() < 1e-12);
    assert!(row["ttfb_ms"].as_i64().unwrap() >= 0);
    assert!(row["duration_ms"].as_i64().unwrap() >= 0);
    assert!(row["error"].is_null());
}

#[tokio::test]
async fn injects_include_usage_and_logs_streaming_usage() {
    let mock = MockServer::start().await;
    let sse_body = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
                    data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                    data: {\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":7}}\n\n\
                    data: [DONE]\n\n";
    // The router must have injected include_usage into the forwarded body.
    let expected_body = serde_json::json!({
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(wiremock::matchers::body_json(&expected_body))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse_body, "text/event-stream"),
        )
        .expect(1)
        .mount(&mock)
        .await;

    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _id) = router.create_token().await;

    let stream_req =
        r#"{"model":"gpt-test","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(stream_req)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert_eq!(text, sse_body, "SSE must be relayed byte-for-byte");

    let rows = router.wait_for_logs(1).await;
    let row = &rows[0];
    assert_eq!(row["streamed"], true);
    assert_eq!(row["prompt_tokens"], 42);
    assert_eq!(row["completion_tokens"], 7);
    assert_eq!(row["total_tokens"], 49);
    assert_eq!(row["finish_reason"], "stop");
    assert!(
        row["cost"].is_null(),
        "no pricing configured for this model"
    );
}

#[tokio::test]
async fn logs_anthropic_streaming_usage_with_cache_normalization() {
    let mock = MockServer::start().await;
    let sse_body = "event: message_start\n\
                    data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":30,\"output_tokens\":1}}}\n\n\
                    event: content_block_delta\n\
                    data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n\
                    event: message_delta\n\
                    data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":12}}\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse_body, "text/event-stream"),
        )
        .mount(&mock)
        .await;

    let config = format!(
        "[[backends]]\nname = \"a1\"\nkind = \"anthropic\"\nbase_url = {:?}\napi_key = \"k\"\n",
        mock.uri()
    ) + "[[models]]\nname = \"claude-test\"\n\
         [models.pricing]\ninput_per_mtok = 3.0\noutput_per_mtok = 15.0\ncached_input_per_mtok = 0.3\n\
         [[models.targets]]\nbackend = \"a1\"\n";
    let router = TestRouter::start(&config).await;
    let (token, _id) = router.create_token().await;

    let body = r#"{"model":"claude-test","max_tokens":100,"messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let resp = router
        .client
        .post(router.url("/v1/messages"))
        .header("x-api-key", &token)
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert_eq!(text, sse_body, "anthropic streams are never modified");

    let rows = router.wait_for_logs(1).await;
    let row = &rows[0];
    assert_eq!(row["streamed"], true);
    // input_tokens excludes cache reads upstream; normalized prompt includes them.
    assert_eq!(row["prompt_tokens"], 40);
    assert_eq!(row["completion_tokens"], 12);
    assert_eq!(row["cached_tokens"], 30);
    assert_eq!(row["total_tokens"], 52);
    assert_eq!(row["finish_reason"], "end_turn");
    let expected = 10.0 / 1e6 * 3.0 + 30.0 / 1e6 * 0.3 + 12.0 / 1e6 * 15.0;
    assert!((row["cost"].as_f64().unwrap() - expected).abs() < 1e-12);
}

#[tokio::test]
async fn logs_failure_rows_for_unknown_model_and_unreachable_backend() {
    let router = TestRouter::start(
        &(backend_toml("b1", "http://127.0.0.1:1", "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _id) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(r#"{"model":"nope"}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);

    let rows = router.wait_for_logs(2).await;
    let unknown = rows.iter().find(|r| r["model"] == "nope").unwrap();
    assert_eq!(unknown["status"], 404);
    assert!(
        unknown["error"]
            .as_str()
            .unwrap()
            .contains("not configured")
    );
    assert!(unknown["backend"].is_null());

    let unreachable = rows.iter().find(|r| r["model"] == "gpt-test").unwrap();
    assert_eq!(unreachable["status"], 502);
    assert!(unreachable["error"].as_str().unwrap().contains("failover"));
    assert!(unreachable["backend"].is_null());
}

#[tokio::test]
async fn admin_log_and_usage_queries() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        })))
        .mount(&mock)
        .await;

    let config = backend_toml("b1", &mock.uri(), "k")
        + "[[models]]\nname = \"gpt-test\"\n\
           [models.pricing]\ninput_per_mtok = 1.0\noutput_per_mtok = 2.0\n\
           [[models.targets]]\nbackend = \"b1\"\n"
        + "[[models]]\nname = \"other-model\"\n[[models.targets]]\nbackend = \"b1\"\n";
    let router = TestRouter::start(&config).await;
    let (token, id) = router.create_token().await;

    for model in ["gpt-test", "gpt-test", "other-model"] {
        let body = format!(r#"{{"model":"{model}","messages":[]}}"#);
        let resp = router
            .client
            .post(router.url("/v1/chat/completions"))
            .bearer_auth(&token)
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        resp.bytes().await.unwrap();
    }
    router.wait_for_logs(3).await;

    // Filtered list: only gpt-test rows.
    let page: serde_json::Value = router
        .client
        .get(router.url("/admin/logs?model=gpt-test"))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(page["items"].as_array().unwrap().len(), 2);
    assert!(page["next_cursor"].is_null());
    let first = &page["items"][0];
    assert_eq!(first["model"], "gpt-test");
    assert_eq!(first["prompt_tokens"], 10);
    assert!(
        first["created_at"].as_str().unwrap().contains('T'),
        "RFC3339 timestamp"
    );

    // Keyset pagination: page size 2 over 3 rows.
    let page1: serde_json::Value = router
        .client
        .get(router.url("/admin/logs?limit=2"))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(page1["items"].as_array().unwrap().len(), 2);
    let cursor = page1["next_cursor"].as_str().unwrap();
    let page2: serde_json::Value = router
        .client
        .get(router.url(&format!("/admin/logs?limit=2&cursor={cursor}")))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(page2["items"].as_array().unwrap().len(), 1);
    let mut all_ids: Vec<String> = page1["items"]
        .as_array()
        .unwrap()
        .iter()
        .chain(page2["items"].as_array().unwrap())
        .map(|r| r["id"].as_str().unwrap().to_string())
        .collect();
    all_ids.sort();
    all_ids.dedup();
    assert_eq!(all_ids.len(), 3, "pages must not overlap");

    // Log detail round-trip and 404.
    let log_id = page1["items"][0]["id"].as_str().unwrap();
    let detail: serde_json::Value = router
        .client
        .get(router.url(&format!("/admin/logs/{log_id}")))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(detail["id"], *log_id);
    let resp = router
        .client
        .get(router.url("/admin/logs/nope"))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // Usage grouped by model.
    let usage: serde_json::Value = router
        .client
        .get(router.url("/admin/usage?group_by=model"))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let buckets = usage.as_array().unwrap();
    let gpt = buckets.iter().find(|b| b["bucket"] == "gpt-test").unwrap();
    assert_eq!(gpt["requests"], 2);
    assert_eq!(gpt["prompt_tokens"], 20);
    assert_eq!(gpt["completion_tokens"], 10);
    let expected_cost = 2.0 * (10.0 / 1e6 * 1.0 + 5.0 / 1e6 * 2.0);
    assert!((gpt["cost"].as_f64().unwrap() - expected_cost).abs() < 1e-12);

    // Usage grouped by day contains today's bucket with all 3 requests.
    let usage: serde_json::Value = router
        .client
        .get(router.url("/admin/usage"))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let day = &usage.as_array().unwrap()[0];
    assert_eq!(day["requests"], 3);
    assert!(day["bucket"].as_str().unwrap().len() == 10, "YYYY-MM-DD");

    // Key detail includes aggregate cards.
    let detail: serde_json::Value = router
        .client
        .get(router.url(&format!("/admin/tokens/{id}")))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(detail["usage"]["today"]["requests"], 3);
    assert_eq!(detail["usage"]["all_time"]["prompt_tokens"], 30);
    assert_eq!(detail["usage"]["this_month"]["completion_tokens"], 15);

    // group_by validation.
    let resp = router
        .client
        .get(router.url("/admin/usage?group_by=hour"))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

/// The admin chat console: it dispatches through the real proxy path,
/// pinned to the chosen backend, and logs as the synthetic admin key.
#[tokio::test]
async fn admin_chat_console_dispatches_through_the_proxy() {
    let good = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer real-key"))
        .and(body_json_string(
            r#"{"model":"upstream-name","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"ping"}],"max_tokens":16}"#,
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3}
        })))
        .mount(&good)
        .await;

    let broken = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_json(serde_json::json!({
            "error": {"message": "upstream is down"}
        })))
        .mount(&broken)
        .await;

    // One model on two backends: the console must reach whichever it names.
    let config = backend_toml("good", &good.uri(), "real-key")
        + &backend_toml("broken", &broken.uri(), "k")
        + "[[models]]\nname = \"gpt-test\"\n\
           [models.pricing]\ninput_per_mtok = 1.0\noutput_per_mtok = 2.0\n\
           [[models.targets]]\nbackend = \"good\"\nmodel = \"upstream-name\"\n\
           [[models.targets]]\nbackend = \"broken\"\n";
    let router = TestRouter::start(&config).await;

    let ask = |backend: &str| {
        serde_json::json!({
            "backend": backend,
            "model": "gpt-test",
            "path": "/v1/chat/completions",
            "system": "be brief",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 16,
        })
    };

    let (status, body) = router
        .admin(
            reqwest::Method::POST,
            "/admin/playground",
            Some(ask("good")),
        )
        .await;
    assert_eq!(status, 200);
    assert_eq!(body["status"], 200);
    assert_eq!(body["text"], "pong", "reply text is extracted per dialect");
    assert_eq!(body["body"]["choices"][0]["message"]["content"], "pong");
    let request_id = body["request_id"].as_str().unwrap().to_string();

    // Pinning holds: naming the broken backend surfaces its 503 instead of
    // failing over to the healthy target of the same model.
    let (status, body) = router
        .admin(
            reqwest::Method::POST,
            "/admin/playground",
            Some(ask("broken")),
        )
        .await;
    assert_eq!(status, 200, "a failed test is a result, not a fault");
    assert_eq!(body["status"], 503);
    assert_eq!(body["text"], "upstream is down");

    // Both calls went through the normal logging path, attributed to the
    // console's synthetic key and to the backend each pinned.
    let rows = router.wait_for_logs(2).await;
    let first = rows
        .iter()
        .find(|r| r["id"] == request_id.as_str())
        .unwrap();
    assert_eq!(first["token_id"], "admin");
    assert_eq!(first["backend"], "good");
    assert_eq!(first["upstream_model"], "upstream-name");
    assert_eq!(first["prompt_tokens"], 12);
    assert_eq!(
        first["cost"].as_f64().unwrap(),
        12.0 / 1e6 + 3.0 * 2.0 / 1e6
    );
    assert!(
        rows.iter()
            .any(|r| r["backend"] == "broken" && r["status"] == 503)
    );

    // The console key has no `tokens` row, so its name is filled in.
    let (_, page) = router
        .admin(reqwest::Method::GET, "/admin/logs?token_id=admin", None)
        .await;
    assert_eq!(page["items"].as_array().unwrap().len(), 2);
    assert_eq!(page["items"][0]["token_name"], "admin console");
    let (_, usage) = router
        .admin(reqwest::Method::GET, "/admin/usage?group_by=token", None)
        .await;
    assert_eq!(usage[0]["bucket"], "admin");
    assert_eq!(usage[0]["label"], "admin console");

    // Pairings the catalog cannot serve are rejected as bad requests.
    let mut wrong = ask("good");
    wrong["path"] = serde_json::json!("/v1/messages");
    let (status, err) = router
        .admin(reqwest::Method::POST, "/admin/playground", Some(wrong))
        .await;
    assert_eq!(status, 400);
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no target on backend")
    );

    let mut unknown = ask("good");
    unknown["model"] = serde_json::json!("nope");
    assert_eq!(
        router
            .admin(reqwest::Method::POST, "/admin/playground", Some(unknown))
            .await
            .0,
        404
    );

    // And it stays behind the admin token.
    let resp = router
        .client
        .post(router.url("/admin/playground"))
        .json(&ask("good"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

/// The console's transcription mode: base64 audio in, a real multipart form
/// out through the proxy, and diarized segments rendered one per line.
#[tokio::test]
async fn admin_console_transcribes_through_the_proxy() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(header("authorization", "Bearer k"))
        .and(|req: &Request| {
            let ct = req.headers.get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("");
            let body = String::from_utf8_lossy(&req.body);
            ct.starts_with("multipart/form-data; boundary=")
                && body.contains("filename=\"call.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFdata")
                && body.contains("name=\"model\"\r\n\r\ngpt-4o-transcribe-diarize\r\n")
                && body.contains("name=\"language\"\r\n\r\nen\r\n")
                && body.contains("name=\"response_format\"\r\n\r\ndiarized_json\r\n")
        })
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "text": "hi there",
            "segments": [
                {"type": "transcript.text.segment", "id": "seg_0", "speaker": "A", "start": 0.0, "end": 0.8, "text": "hi"},
                {"type": "transcript.text.segment", "id": "seg_1", "speaker": "B", "start": 0.9, "end": 1.5, "text": "there"}
            ],
            "usage": {"type": "tokens", "input_tokens": 20, "output_tokens": 4, "total_tokens": 24}
        })))
        .mount(&mock)
        .await;

    let config = backend_toml("b1", &mock.uri(), "k")
        + "[[models]]\nname = \"diarize\"\n[[models.targets]]\nbackend = \"b1\"\nmodel = \"gpt-4o-transcribe-diarize\"\n";
    let router = TestRouter::start(&config).await;

    let (status, body) = router
        .admin(
            reqwest::Method::POST,
            "/admin/playground",
            Some(serde_json::json!({
                "backend": "b1",
                "model": "diarize",
                "path": "/v1/audio/transcriptions",
                "audio": {"filename": "call.wav", "content_type": "audio/wav", "data": "UklGRmRhdGE="},
                "language": "en",
                "response_format": "diarized_json",
            })),
        )
        .await;
    assert_eq!(status, 200);
    assert_eq!(body["status"], 200, "{body}");
    assert_eq!(body["text"], "[0.0s] A: hi\n[0.9s] B: there");

    let rows = router.wait_for_logs(1).await;
    assert_eq!(rows[0]["token_id"], "admin");
    assert_eq!(rows[0]["path"], "/v1/audio/transcriptions");
    assert_eq!(rows[0]["prompt_tokens"], 20);
    assert_eq!(rows[0]["completion_tokens"], 4);
}

/// The model/provider analytics surface: grouping by backend, the two
/// detail endpoints and their breakdowns, and the logs backend filter.
#[tokio::test]
async fn model_and_provider_analytics() {
    let ok = || {
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }))
    };
    let (m1, m2) = (MockServer::start().await, MockServer::start().await);
    for mock in [&m1, &m2] {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ok())
            .mount(mock)
            .await;
    }

    let config = backend_toml("b1", &m1.uri(), "k")
        + &backend_toml("b2", &m2.uri(), "k")
        + "[[models]]\nname = \"gpt-test\"\n\
           [models.pricing]\ninput_per_mtok = 1.0\noutput_per_mtok = 2.0\n\
           [[models.targets]]\nbackend = \"b1\"\n"
        + "[[models]]\nname = \"claude-test\"\n[[models.targets]]\nbackend = \"b2\"\n";
    let router = TestRouter::start(&config).await;

    let (tok_a, id_a) = router.create_token().await;
    let (_, second) = router
        .admin(
            reqwest::Method::POST,
            "/admin/tokens",
            Some(serde_json::json!({"name": "second-key"})),
        )
        .await;
    let tok_b = second["token"].as_str().unwrap().to_string();

    // b1 serves gpt-test for both keys; b2 serves claude-test once. The
    // unknown model is rejected before dispatch, so it has no backend.
    for (token, model) in [
        (&tok_a, "gpt-test"),
        (&tok_a, "gpt-test"),
        (&tok_a, "claude-test"),
        (&tok_b, "gpt-test"),
        (&tok_a, "nope-model"),
    ] {
        router
            .client
            .post(router.url("/v1/chat/completions"))
            .bearer_auth(token)
            .body(format!(r#"{{"model":"{model}","messages":[]}}"#))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
    }
    router.wait_for_logs(5).await;

    // Grouped by backend: one bucket per provider, plus a null bucket for
    // the request that never reached one.
    let (_, usage) = router
        .admin(reqwest::Method::GET, "/admin/usage?group_by=backend", None)
        .await;
    let buckets = usage.as_array().unwrap();
    let b1 = buckets.iter().find(|b| b["bucket"] == "b1").unwrap();
    assert_eq!(b1["requests"], 3);
    assert_eq!(b1["prompt_tokens"], 30);
    assert_eq!(b1["errors"], 0);
    assert!(
        b1["label"].is_null(),
        "backend buckets carry no separate label"
    );
    assert_eq!(
        buckets.iter().find(|b| b["bucket"] == "b2").unwrap()["requests"],
        1
    );
    let unrouted = buckets.iter().find(|b| b["bucket"].is_null()).unwrap();
    assert_eq!(unrouted["requests"], 1);
    assert_eq!(unrouted["errors"], 1);
    // Priced traffic ranks first.
    assert_eq!(buckets[0]["bucket"], "b1");

    // Grouped by token: key ids with their names as labels.
    let (_, usage) = router
        .admin(reqwest::Method::GET, "/admin/usage?group_by=token", None)
        .await;
    let a = usage
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["bucket"] == id_a.as_str())
        .unwrap();
    assert_eq!(a["label"], "test");
    assert_eq!(a["requests"], 4);
    assert_eq!(a["errors"], 1);

    // Model detail: who served it and who called it.
    let (_, detail) = router
        .admin(reqwest::Method::GET, "/admin/models/gpt-test/usage", None)
        .await;
    assert_eq!(detail["model"], "gpt-test");
    assert_eq!(detail["usage"]["all_time"]["requests"], 3);
    assert_eq!(detail["usage"]["today"]["prompt_tokens"], 30);
    let by_backend = detail["by_backend"].as_array().unwrap();
    assert_eq!(by_backend.len(), 1);
    assert_eq!(by_backend[0]["bucket"], "b1");
    assert_eq!(by_backend[0]["requests"], 3);
    let by_token = detail["by_token"].as_array().unwrap();
    assert_eq!(by_token.len(), 2, "both keys called this model");
    assert!(
        by_token
            .iter()
            .any(|t| t["label"] == "second-key" && t["requests"] == 1)
    );
    assert_eq!(
        detail["daily"].as_array().unwrap().len(),
        1,
        "one day of traffic"
    );

    // Provider detail: the models it carried.
    let (_, detail) = router
        .admin(reqwest::Method::GET, "/admin/backends/b2/usage", None)
        .await;
    assert_eq!(detail["backend"], "b2");
    assert_eq!(detail["usage"]["all_time"]["requests"], 1);
    let by_model = detail["by_model"].as_array().unwrap();
    assert_eq!(by_model.len(), 1);
    assert_eq!(by_model[0]["bucket"], "claude-test");
    assert_eq!(detail["by_token"][0]["label"], "test");

    // Names outside the catalog answer with their history rather than 404 —
    // logs outlive the models that produced them.
    let (status, detail) = router
        .admin(reqwest::Method::GET, "/admin/models/nope-model/usage", None)
        .await;
    assert_eq!(status, 200);
    assert_eq!(detail["usage"]["all_time"]["errors"], 1);
    assert!(detail["by_backend"][0]["bucket"].is_null());

    // Logs filter by provider.
    let (_, page) = router
        .admin(reqwest::Method::GET, "/admin/logs?backend=b2", None)
        .await;
    let items = page["items"].as_array().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["model"], "claude-test");
}

#[tokio::test]
async fn credit_and_token_limits_are_enforced() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500}
        })))
        .mount(&mock)
        .await;

    // gpt-test costs 0.002 + 0.005 = 0.007 per request; free-model has no
    // pricing so only token limits can bound it (the vllm case).
    let config = backend_toml("b1", &mock.uri(), "k")
        + "[[models]]\nname = \"gpt-test\"\n\
           [models.pricing]\ninput_per_mtok = 2.0\noutput_per_mtok = 10.0\n\
           [[models.targets]]\nbackend = \"b1\"\n"
        + "[[models]]\nname = \"free-model\"\n[[models.targets]]\nbackend = \"b1\"\n";
    let router = TestRouter::start(&config).await;

    // Key 1: credit limit below one request's cost.
    let resp = router
        .client
        .post(router.url("/admin/tokens"))
        .bearer_auth(&router.admin_token)
        .json(&serde_json::json!({
            "name": "credit-limited",
            "credit_limit": 0.005,
            "credit_limit_reset": "monthly"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: serde_json::Value = resp.json().await.unwrap();
    let credit_token = body["token"].as_str().unwrap().to_string();
    let credit_id = body["id"].as_str().unwrap().to_string();

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&credit_token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "first request is under the limit");
    resp.bytes().await.unwrap();
    router.wait_for_logs(1).await; // accrual happens before the row is queued

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&credit_token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 429);
    let err: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(err["error"]["type"], "insufficient_quota");
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("credit limit")
    );

    // The rejection itself is logged.
    let rows = router.wait_for_logs(2).await;
    let rejected = rows.iter().find(|r| r["status"] == 429).unwrap();
    assert_eq!(rejected["token_id"], credit_id);

    // Key 2: token limit on the unpriced model (1500 tokens per request).
    let resp = router
        .client
        .post(router.url("/admin/tokens"))
        .bearer_auth(&router.admin_token)
        .json(&serde_json::json!({
            "name": "token-limited",
            "token_limit": 2000,
            "token_limit_reset": "daily"
        }))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    let tok_token = body["token"].as_str().unwrap().to_string();
    let tok_id = body["id"].as_str().unwrap().to_string();

    let free_body = r#"{"model":"free-model","messages":[]}"#;
    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&tok_token)
        .body(free_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    resp.bytes().await.unwrap();
    router.wait_for_logs(3).await;

    // 1500 < 2000: still admitted; accrues to 3000.
    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&tok_token)
        .body(free_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    resp.bytes().await.unwrap();
    router.wait_for_logs(4).await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&tok_token)
        .body(free_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 429);
    let err: serde_json::Value = resp.json().await.unwrap();
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("token limit")
    );

    // The key list shows current-window usage for the progress bars.
    let list: serde_json::Value = router
        .client
        .get(router.url("/admin/tokens"))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let tok_entry = list
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["id"] == tok_id.as_str())
        .unwrap();
    assert_eq!(tok_entry["usage"]["window_tokens"], 3000);

    // Raising the limit via PATCH admits requests again immediately.
    let resp = router
        .client
        .patch(router.url(&format!("/admin/tokens/{tok_id}")))
        .bearer_auth(&router.admin_token)
        .json(&serde_json::json!({"token_limit": 10000}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&tok_token)
        .body(free_body)
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "re-warmed window (3000) is under the raised limit"
    );
}

#[tokio::test]
async fn separate_admin_listener_splits_surfaces() {
    let (router, admin_addr) = TestRouter::start_with_admin_listener(
        &(backend_toml("b1", "http://127.0.0.1:1", "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let admin_url = |p: &str| format!("http://{admin_addr}{p}");

    // Admin routes are absent from the main listener.
    let resp = router
        .client
        .get(router.url("/admin/tokens"))
        .bearer_auth(&router.admin_token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // The admin listener serves them (with auth still required).
    let resp = router
        .client
        .get(admin_url("/admin/tokens"))
        .bearer_auth("wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
    let resp = router
        .client
        .post(admin_url("/admin/tokens"))
        .bearer_auth(&router.admin_token)
        .json(&serde_json::json!({"name": "split"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: serde_json::Value = resp.json().await.unwrap();
    let token = body["token"].as_str().unwrap();

    // Proxy routes are absent from the admin listener.
    let resp = router
        .client
        .post(admin_url("/v1/chat/completions"))
        .bearer_auth(token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // The main listener proxies with the key created via the admin port
    // (502: the fake backend is unreachable, but auth + routing passed).
    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);

    // Both listeners expose healthz.
    for url in [router.url("/healthz"), admin_url("/healthz")] {
        let resp = router.client.get(url).send().await.unwrap();
        assert_eq!(resp.status(), 200);
    }
}

#[tokio::test]
async fn admin_ui_is_served_without_auth() {
    let router = TestRouter::start(
        &(backend_toml("b1", "http://127.0.0.1:1", "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;

    // No Authorization header: the page itself is public, data is not.
    let resp = router
        .client
        .get(router.url("/admin/ui"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/html")
    );
    // The page is embedded at build time, so it must never be cached: a
    // browser holding the previous deploy's HTML against the new binary's
    // API looks like a missing feature, not a stale page.
    assert_eq!(
        resp.headers()["cache-control"],
        "no-store",
        "the embedded UI must not be heuristically cacheable across a redeploy"
    );
    let body = resp.text().await.unwrap();
    assert!(body.contains("llm-router admin"));

    // /admin redirects to the page.
    let resp = router
        .client
        .get(router.url("/admin"))
        .header("accept", "text/html")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "redirect followed to the UI");
    assert!(resp.url().path().ends_with("/admin/ui"));

    // The split admin listener serves the UI too.
    let (router, admin_addr) = TestRouter::start_with_admin_listener(
        &(backend_toml("b1", "http://127.0.0.1:1", "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let resp = router
        .client
        .get(format!("http://{admin_addr}/admin/ui"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // ...and it is absent from the main listener.
    let resp = router
        .client
        .get(router.url("/admin/ui"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

// ---- catalog (backends + models) admin API -----------------------------

impl TestRouter {
    /// Sends an admin request with a JSON body (omit for GET/DELETE) and
    /// returns the status plus the parsed body (`null` for 204).
    async fn admin(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> (u16, serde_json::Value) {
        let mut req = self
            .client
            .request(method, self.url(path))
            .bearer_auth(&self.admin_token);
        if let Some(b) = body {
            req = req.json(&b);
        }
        let resp = req.send().await.unwrap();
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap();
        let value = serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
        (status, value)
    }
}

/// Backend that echoes which mock answered, so a re-pointed route is
/// visible in the proxied response body.
async fn echo_backend(name: &'static str) -> MockServer {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"content": name}, "finish_reason": "stop"}]
        })))
        .mount(&mock)
        .await;
    mock
}

#[tokio::test]
async fn catalog_api_reroutes_traffic_without_a_restart() {
    let a = echo_backend("from-a").await;
    let b = echo_backend("from-b").await;
    let router = TestRouter::start(
        &(backend_toml("b1", &a.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    let chat = || async {
        let resp = router
            .client
            .post(router.url("/v1/chat/completions"))
            .bearer_auth(&token)
            .body(CHAT_BODY)
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        (status, resp.text().await.unwrap())
    };
    assert!(chat().await.1.contains("from-a"));

    // Add a second backend and re-point the model at it.
    let (status, created) = router
        .admin(
            reqwest::Method::POST,
            "/admin/backends",
            Some(serde_json::json!({
                "name": "b2", "kinds": ["openai"], "base_url": b.uri(), "api_key": "sk-b2"
            })),
        )
        .await;
    assert_eq!(status, 201);
    assert_eq!(created["has_api_key"], true);
    assert!(
        created.get("api_key").is_none(),
        "the stored key is never echoed: {created}"
    );

    let (status, listed) = router
        .admin(reqwest::Method::GET, "/admin/backends", None)
        .await;
    assert_eq!(status, 200);
    assert_eq!(listed.as_array().unwrap().len(), 2);
    assert!(!listed.to_string().contains("sk-b2"));

    let (status, _) = router
        .admin(
            reqwest::Method::PUT,
            "/admin/models/gpt-test",
            Some(serde_json::json!({"targets": [{"backend": "b2", "model": "renamed"}]})),
        )
        .await;
    assert_eq!(status, 200);
    // The next request routes to the new backend — no restart involved.
    assert!(chat().await.1.contains("from-b"));

    // A backend a model still points at cannot be deleted.
    let (status, err) = router
        .admin(reqwest::Method::DELETE, "/admin/backends/b2", None)
        .await;
    assert_eq!(status, 409);
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("gpt-test")
    );

    // Validation mirrors the config file's rules.
    let (status, _) = router
        .admin(
            reqwest::Method::POST,
            "/admin/models",
            Some(serde_json::json!({"name": "x", "targets": [{"backend": "nope"}]})),
        )
        .await;
    assert_eq!(status, 400);
    let (status, _) = router
        .admin(
            reqwest::Method::POST,
            "/admin/models",
            Some(serde_json::json!({"name": "gpt-test", "targets": [{"backend": "b1"}]})),
        )
        .await;
    assert_eq!(status, 409);
    let (status, _) = router
        .admin(
            reqwest::Method::POST,
            "/admin/backends",
            Some(serde_json::json!({
                "name": "b3", "kinds": ["openai"], "base_url": "http://x", "api_key_env": "LLM_ROUTER_SURELY_UNSET"
            })),
        )
        .await;
    assert_eq!(status, 400, "an unset api_key_env is caught at write time");

    // Deleting the model frees the backend and stops routing for the name.
    assert_eq!(
        router
            .admin(reqwest::Method::DELETE, "/admin/models/gpt-test", None)
            .await
            .0,
        204
    );
    assert_eq!(chat().await.0, 404);
    assert_eq!(
        router
            .admin(reqwest::Method::DELETE, "/admin/backends/b2", None)
            .await
            .0,
        204
    );
    assert_eq!(
        router
            .admin(reqwest::Method::DELETE, "/admin/backends/b2", None)
            .await
            .0,
        404
    );
}

// ---- concurrency slots -------------------------------------------------

/// A backend that takes `delay_ms` to answer, so overlapping requests are
/// distinguishable from serialized ones by wall clock alone.
async fn slow_backend(delay_ms: u64) -> MockServer {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(std::time::Duration::from_millis(delay_ms))
                .set_body_json(serde_json::json!({
                    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                })),
        )
        .mount(&mock)
        .await;
    mock
}

#[tokio::test]
async fn max_concurrency_serializes_requests_on_a_backend() {
    const DELAY: u64 = 300;
    let mock = slow_backend(DELAY).await;
    let model = "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n";

    let two_at_once = |router: TestRouter| async move {
        let (token, _) = router.create_token().await;
        let call = || {
            router
                .client
                .post(router.url("/v1/chat/completions"))
                .bearer_auth(&token)
                .body(CHAT_BODY)
                .send()
        };
        let started = std::time::Instant::now();
        let (a, b) = tokio::join!(call(), call());
        assert_eq!(a.unwrap().status(), 200);
        assert_eq!(b.unwrap().status(), 200);
        started.elapsed()
    };

    // Control: unlimited, so the two requests overlap on the backend.
    let unlimited = TestRouter::start(&(backend_toml("b1", &mock.uri(), "k") + model)).await;
    let overlapped = two_at_once(unlimited).await;
    assert!(
        overlapped < std::time::Duration::from_millis(DELAY * 2),
        "unlimited requests should overlap, took {overlapped:?}"
    );

    // With one slot the second request waits for the first to finish.
    let limited = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k") + "max_concurrency = 1\n" + model),
    )
    .await;
    let serialized = two_at_once(limited).await;
    assert!(
        serialized >= std::time::Duration::from_millis(DELAY * 2),
        "max_concurrency = 1 should serialize, took {serialized:?}"
    );
}

#[tokio::test]
async fn a_client_disconnecting_mid_stream_releases_the_slot() {
    let mock = MockServer::start().await;
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
               data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4}}\n\n\
               data: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse, "text/event-stream"),
        )
        .mount(&mock)
        .await;

    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k")
            + "max_concurrency = 1\n"
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;
    let stream_body =
        r#"{"model":"gpt-test","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

    // Take the only slot, then walk away without reading the body — the
    // shape a client disconnect takes. Nothing releases the slot
    // explicitly; the tee's Drop is the only path.
    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(stream_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    drop(resp);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // A leaked slot would hang this forever rather than fail it.
    let next = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        router
            .client
            .post(router.url("/v1/chat/completions"))
            .bearer_auth(&token)
            .body(CHAT_BODY)
            .send(),
    )
    .await
    .expect("the disconnected request must have released its slot")
    .unwrap();
    assert_eq!(next.status(), 200);

    // The registry entry rides in the same LogContext, so the drop that
    // released the slot must have deregistered the request too — a leak
    // here would pin a phantom row in the logs page's active panel.
    let (_, page) = router
        .admin(reqwest::Method::GET, "/admin/active", None)
        .await;
    assert_eq!(
        page["count"], 0,
        "a disconnected request stayed in the registry: {page}"
    );
}

#[tokio::test]
async fn failing_over_off_a_5xx_backend_releases_its_slot() {
    let broken = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string("upstream down"))
        .mount(&broken)
        .await;
    let healthy = echo_backend("from-healthy").await;

    // b1 carries the heavier weight, so it is the primary pick — and so
    // acquires its single slot — on most requests. Each of those must give
    // the slot back before trying b2.
    let router = TestRouter::start(
        &(backend_toml("b1", &broken.uri(), "k")
            + "max_concurrency = 1\n"
            + &backend_toml("b2", &healthy.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\n\
               [[models.targets]]\nbackend = \"b1\"\nweight = 3\n\
               [[models.targets]]\nbackend = \"b2\"\nweight = 1\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    for i in 0..4 {
        let resp = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            router
                .client
                .post(router.url("/v1/chat/completions"))
                .bearer_auth(&token)
                .body(CHAT_BODY)
                .send(),
        )
        .await
        .unwrap_or_else(|_| panic!("request {i} blocked — the 5xx hop leaked b1's slot"))
        .unwrap();
        assert_eq!(resp.status(), 200);
        assert!(resp.text().await.unwrap().contains("from-healthy"));
    }

    let rows = router.wait_for_logs(4).await;
    // Every request that started on b1 records the failover and still ends
    // up served (and billed) by b2.
    assert!(rows.iter().all(|r| r["backend"] == "b2"));
    assert!(rows.iter().any(|r| {
        r["error"]
            .as_str()
            .is_some_and(|e| e.contains("b1: HTTP 503"))
    }));
}

// ---- exclusive groups (end to end) -------------------------------------

/// Stands in for one accelerator hosting several inference servers, and
/// records exactly what the group exists to prevent: two of them generating
/// at the same time. wiremock cannot do this — its responders are
/// synchronous and its delay is applied outside them, so nothing observes
/// the window a request is actually resident for.
#[derive(Default)]
struct DeviceWatch {
    live: std::sync::Mutex<std::collections::HashMap<String, usize>>,
    violations: std::sync::Mutex<Vec<String>>,
}

/// Serves `/<engine>/v1/chat/completions` for each named engine, holding
/// each request for `hold_ms` while it counts as resident.
async fn device_server(
    engines: &[&'static str],
    hold_ms: u64,
) -> (SocketAddr, std::sync::Arc<DeviceWatch>) {
    use axum::extract::Path as AxumPath;
    use axum::routing::post;

    let watch = std::sync::Arc::new(DeviceWatch::default());
    let known: std::collections::HashSet<String> = engines.iter().map(|e| e.to_string()).collect();
    let state = (std::sync::Arc::clone(&watch), known);

    let app = axum::Router::new()
        .route(
            "/{engine}/v1/chat/completions",
            post(
                move |axum::extract::State((watch, known)): axum::extract::State<(
                    std::sync::Arc<DeviceWatch>,
                    std::collections::HashSet<String>,
                )>,
                      AxumPath(engine): AxumPath<String>| async move {
                    assert!(known.contains(&engine), "unexpected engine {engine}");
                    {
                        let mut live = watch.live.lock().unwrap();
                        *live.entry(engine.clone()).or_default() += 1;
                        let busy: Vec<String> = live
                            .iter()
                            .filter(|(_, n)| **n > 0)
                            .map(|(k, _)| k.clone())
                            .collect();
                        if busy.len() > 1 {
                            watch.violations.lock().unwrap().push(busy.join(" + "));
                        }
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(hold_ms)).await;
                    *watch.live.lock().unwrap().get_mut(&engine).unwrap() -= 1;
                    axum::Json(serde_json::json!({
                        "choices": [{"message": {"content": engine}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 5}
                    }))
                },
            ),
        )
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, watch)
}

#[tokio::test]
async fn grouped_backends_never_generate_at_the_same_time() {
    let (device, watch) = device_server(&["moe", "dense"], 60).await;
    let (embed_addr, embed_watch) = device_server(&["embed"], 60).await;

    let grouped = |name: &str, engine: &str, cap: i64| {
        format!(
            "[[backends]]\nname = {name:?}\nkind = \"openai\"\n\
             base_url = \"http://{device}/{engine}\"\napi_key = \"k\"\n\
             scheduling_class = \"grouped\"\nexclusive_group = \"dgx\"\n\
             max_concurrency = 2\nmax_tokens_cap = {cap}\ntok_s_estimate = 90.0\n"
        )
    };
    let config = "[[exclusive_groups]]\nid = \"dgx\"\n\
                  min_hold_ms = 40\nmax_hold_ms = 300\ndrain_timeout_ms = 5000\n"
        .to_string()
        + &grouped("moe-local", "moe", 1024)
        + &grouped("dense-local", "dense", 512)
        // Same device, but exempt: a bounded forward pass interleaves
        // harmlessly and must never wait for the lease.
        + &format!(
            "[[backends]]\nname = \"embed-local\"\nkind = \"openai\"\n\
             base_url = \"http://{embed_addr}/embed\"\napi_key = \"k\"\n\
             scheduling_class = \"exempt\"\nexclusive_group = \"dgx\"\n"
        )
        + "[[models]]\nname = \"qwen-moe\"\n[[models.targets]]\nbackend = \"moe-local\"\n\
           [[models]]\nname = \"qwen-dense\"\n[[models.targets]]\nbackend = \"dense-local\"\n\
           [[models]]\nname = \"embed\"\n[[models.targets]]\nbackend = \"embed-local\"\n";

    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    let call = |model: &'static str| {
        let (client, url, token) = (
            router.client.clone(),
            router.url("/v1/chat/completions"),
            token.clone(),
        );
        async move {
            let resp = client
                .post(url)
                .bearer_auth(&token)
                .body(format!(
                    r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}]}}"#
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            resp.text().await.unwrap()
        }
    };

    // 24 requests alternating between the two large models, all at once,
    // plus embedding traffic that must sail past the whole thing.
    let heavy = (0..24).map(|i| call(if i % 2 == 0 { "qwen-moe" } else { "qwen-dense" }));
    let light = (0..8).map(|_| call("embed"));
    let started = std::time::Instant::now();
    let (heavy, light) = tokio::join!(
        futures_util::future::join_all(heavy),
        futures_util::future::join_all(light),
    );
    let elapsed = started.elapsed();

    assert_eq!(heavy.len(), 24);
    assert_eq!(light.len(), 8);
    assert!(light.iter().all(|b| b.contains("embed")));
    assert!(
        watch.violations.lock().unwrap().is_empty(),
        "grouped backends overlapped: {:?}",
        watch.violations.lock().unwrap()
    );
    assert!(embed_watch.violations.lock().unwrap().is_empty());

    // The embedder ran alongside the contended pair rather than behind it:
    // eight 60 ms requests would take 480 ms serialized, and the whole run
    // took less than the large models' own serialized time.
    assert!(
        elapsed < std::time::Duration::from_secs(20),
        "took {elapsed:?}"
    );

    let rows = router.wait_for_logs(32).await;
    assert_eq!(rows.iter().filter(|r| r["model"] == "embed").count(), 8);
    assert!(rows.iter().all(|r| r["status"] == 200));
}

#[tokio::test]
async fn grouped_targets_clamp_generated_tokens_and_fallbacks_do_not() {
    let local = MockServer::start().await;
    let remote = MockServer::start().await;
    for mock in [&local, &remote] {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "hi"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5}
            })))
            .mount(mock)
            .await;
    }

    let config = "[[exclusive_groups]]\nid = \"dgx\"\n\
                  min_hold_ms = 50\nmax_hold_ms = 200\ndrain_timeout_ms = 5000\n"
        .to_string()
        + &format!(
            "[[backends]]\nname = \"moe-local\"\nkind = \"openai\"\nbase_url = {:?}\n\
             scheduling_class = \"grouped\"\nexclusive_group = \"dgx\"\n\
             max_tokens_cap = 1024\ntok_s_estimate = 90.0\n",
            local.uri()
        )
        + &backend_toml("remote", &remote.uri(), "k")
        + "[[models]]\nname = \"capped\"\n[[models.targets]]\nbackend = \"moe-local\"\n\
           [[models]]\nname = \"uncapped\"\n[[models.targets]]\nbackend = \"remote\"\n";
    let router = TestRouter::start(&config).await;
    let (token, _) = router.create_token().await;

    let call = |body: String| {
        let (client, url, token) = (
            router.client.clone(),
            router.url("/v1/chat/completions"),
            token.clone(),
        );
        async move {
            let resp = client
                .post(url)
                .bearer_auth(&token)
                .body(body)
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
        }
    };

    // Over the cap on a grouped backend: brought down to it.
    call(r#"{"model":"capped","max_tokens":8000,"messages":[]}"#.to_string()).await;
    // No limit at all: the cap is injected, so occupancy stays bounded.
    call(r#"{"model":"capped","messages":[]}"#.to_string()).await;
    // Already under the cap: left exactly as the client wrote it.
    call(r#"{"model":"capped","max_tokens":16,"messages":[]}"#.to_string()).await;
    // The same request to an ungrouped backend keeps its own value.
    call(r#"{"model":"uncapped","max_tokens":8000,"messages":[]}"#.to_string()).await;

    async fn sent(mock: &MockServer) -> Vec<serde_json::Value> {
        mock.received_requests()
            .await
            .unwrap()
            .iter()
            .map(|r| serde_json::from_slice(&r.body).unwrap())
            .collect()
    }
    let local_bodies = sent(&local).await;
    assert_eq!(
        local_bodies[0]["max_tokens"], 1024,
        "over-cap request clamped"
    );
    assert_eq!(
        local_bodies[1]["max_tokens"], 1024,
        "unlimited request capped"
    );
    assert_eq!(
        local_bodies[2]["max_tokens"], 16,
        "under-cap request untouched"
    );

    let remote_bodies = sent(&remote).await;
    assert_eq!(
        remote_bodies[0]["max_tokens"], 8000,
        "a backend outside the group is never clamped"
    );
}

// ---- QoS shedding ------------------------------------------------------

/// A local device held busy, plus a priced remote fallback to shed onto.
/// `hold_ms` is how long the local backend occupies the device.
async fn shed_setup(hold_ms: u64, extra_model_toml: &str) -> (TestRouter, MockServer, MockServer) {
    let local = slow_backend(hold_ms).await;
    let remote = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"content": "from-openrouter"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500}
        })))
        .mount(&remote)
        .await;

    let config = "[[exclusive_groups]]\nid = \"dgx\"\n\
                  min_hold_ms = 5000\nmax_hold_ms = 30000\ndrain_timeout_ms = 45000\n"
        .to_string()
        + &format!(
            "[[backends]]\nname = \"moe-local\"\nkind = \"openai\"\nbase_url = {:?}\n\
             scheduling_class = \"grouped\"\nexclusive_group = \"dgx\"\n\
             max_concurrency = 1\nmax_tokens_cap = 1024\ntok_s_estimate = 90.0\n",
            local.uri()
        )
        + &format!(
            "[[backends]]\nname = \"dense-local\"\nkind = \"openai\"\nbase_url = {:?}\n\
             scheduling_class = \"grouped\"\nexclusive_group = \"dgx\"\n\
             max_concurrency = 1\nmax_tokens_cap = 512\ntok_s_estimate = 25.0\n",
            local.uri()
        )
        + &backend_toml("openrouter", &remote.uri(), "sk-remote")
        + extra_model_toml;
    (TestRouter::start(&config).await, local, remote)
}

#[tokio::test]
async fn a_request_that_would_miss_its_deadline_is_shed_to_the_fallback() {
    let model = "[[models]]\nname = \"qwen-moe\"\n\
                 [models.qos]\nmax_wait_ms = 2000\nbackground_max_wait_ms = 60000\n\
                 [[models.targets]]\nbackend = \"moe-local\"\n\
                   [models.targets.pricing]\n  input_per_mtok = 0.0\n  output_per_mtok = 0.0\n\
                 [[models.targets]]\nbackend = \"openrouter\"\nmodel = \"qwen/qwen3-30b-a3b\"\n\
                 fallback_only = true\n\
                   [models.targets.pricing]\n  input_per_mtok = 0.10\n  output_per_mtok = 0.30\n";
    let (router, _local, remote) = shed_setup(400, model).await;
    let (token, _) = router.create_token().await;

    let call = |headers: Vec<(&'static str, &'static str)>| {
        let (client, url, token) = (
            router.client.clone(),
            router.url("/v1/chat/completions"),
            token.clone(),
        );
        async move {
            let mut req = client.post(url).bearer_auth(&token);
            for (k, v) in headers {
                req = req.header(k, v);
            }
            let resp = req
                .body(r#"{"model":"qwen-moe","messages":[{"role":"user","content":"hi"}]}"#)
                .send()
                .await
                .unwrap();
            (resp.status().as_u16(), resp.text().await.unwrap())
        }
    };

    // First request takes the device — served locally, no fallback.
    let first = tokio::spawn(call(vec![]));
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;

    // Second arrives while the device is held. With min_hold_ms = 5000 and
    // a 2 s deadline the predicted wait cannot fit, so it sheds rather than
    // queueing — the shedder behavior these settings imply.
    let (status, body) = call(vec![]).await;
    assert_eq!(status, 200);
    assert!(body.contains("from-openrouter"), "should have shed: {body}");

    let (status, body) = first.await.unwrap();
    assert_eq!(status, 200);
    assert!(
        !body.contains("from-openrouter"),
        "the first request stays local"
    );

    let rows = router.wait_for_logs(2).await;
    let shed = rows
        .iter()
        .find(|r| r["backend"] == "openrouter")
        .expect("one row should be attributed to the fallback");
    assert_eq!(shed["upstream_model"], "qwen/qwen3-30b-a3b");
    // The shed reason rides the existing failover channel, so it shows up
    // in the logs UI with no schema change.
    let err = shed["error"].as_str().unwrap_or_default();
    assert!(err.contains("moe-local"), "{err}");
    assert!(err.contains("shed"), "{err}");
    // And it is billed at the fallback target's pricing — the whole point
    // of a remote fallback is that it costs real money, so it has to show
    // up in cost reporting.
    let expected = 1000.0 / 1e6 * 0.10 + 500.0 / 1e6 * 0.30;
    assert!(
        (shed["cost"].as_f64().unwrap() - expected).abs() < 1e-12,
        "{shed}"
    );
    let local_row = rows.iter().find(|r| r["backend"] == "moe-local").unwrap();
    assert_eq!(local_row["cost"], 0.0);
    assert!(local_row["error"].is_null());

    // A background request is allowed to wait for local hardware instead.
    let busy = tokio::spawn(call(vec![]));
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    let (status, body) = call(vec![("x-router-qos", "background")]).await;
    assert_eq!(status, 200);
    assert!(
        !body.contains("from-openrouter"),
        "background traffic waits: {body}"
    );
    busy.await.unwrap();

    assert_eq!(
        remote.received_requests().await.unwrap().len(),
        1,
        "only the interactive request should have gone remote"
    );
}

#[tokio::test]
async fn shedding_without_a_fallback_waits_or_errors_as_configured() {
    let model = |on_shed: &str| {
        format!(
            "[[models]]\nname = \"qwen-moe\"\n\
             on_shed_without_fallback = {on_shed:?}\n\
             [models.qos]\nmax_wait_ms = 100\n\
             [[models.targets]]\nbackend = \"moe-local\"\n"
        )
    };

    // "error": the deadline is honored by failing rather than silently
    // dropping it.
    let (router, _local, _remote) = shed_setup(400, &model("error")).await;
    let (token, _) = router.create_token().await;
    let call = |router: &TestRouter, token: String| {
        let (client, url) = (router.client.clone(), router.url("/v1/chat/completions"));
        async move {
            let resp = client
                .post(url)
                .bearer_auth(&token)
                .body(r#"{"model":"qwen-moe","messages":[{"role":"user","content":"hi"}]}"#)
                .send()
                .await
                .unwrap();
            (resp.status().as_u16(), resp.text().await.unwrap())
        }
    };
    let busy = tokio::spawn(call(&router, token.clone()));
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    let (status, body) = call(&router, token.clone()).await;
    assert_eq!(status, 503, "{body}");
    assert!(body.contains("overloaded_error"), "{body}");
    busy.await.unwrap();
    let rows = router.wait_for_logs(2).await;
    let failed = rows.iter().find(|r| r["status"] == 503).unwrap();
    assert!(
        failed["error"].as_str().unwrap().contains("shed"),
        "{failed}"
    );

    // "wait" (the default): the request queues past its deadline rather
    // than being dropped, and is served locally.
    let (router, _local, _remote) = shed_setup(400, &model("wait")).await;
    let (token, _) = router.create_token().await;
    let busy = tokio::spawn(call(&router, token.clone()));
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    let (status, _) = call(&router, token.clone()).await;
    assert_eq!(status, 200);
    busy.await.unwrap();
    let rows = router.wait_for_logs(2).await;
    assert!(rows.iter().all(|r| r["backend"] == "moe-local"));
    assert!(
        rows.iter().any(|r| r["error"]
            .as_str()
            .is_some_and(|e| e.contains("no fallback target"))),
        "the missed deadline is still recorded: {rows:?}"
    );
}

#[tokio::test]
async fn shedding_skips_the_other_member_of_the_same_device() {
    // Both local targets sit on one device, so shedding moe-local ->
    // dense-local would be pointless — they contend for the same lease.
    // The request must go straight to the remote fallback.
    let model = "[[models]]\nname = \"qwen\"\n\
                 [models.qos]\nmax_wait_ms = 100\n\
                 [[models.targets]]\nbackend = \"moe-local\"\n\
                 [[models.targets]]\nbackend = \"dense-local\"\n\
                 [[models.targets]]\nbackend = \"openrouter\"\nfallback_only = true\n";
    let (router, _local, remote) = shed_setup(400, model).await;
    let (token, _) = router.create_token().await;

    let call = || {
        let (client, url, token) = (
            router.client.clone(),
            router.url("/v1/chat/completions"),
            token.clone(),
        );
        async move {
            let resp = client
                .post(url)
                .bearer_auth(&token)
                .body(r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}]}"#)
                .send()
                .await
                .unwrap();
            (resp.status().as_u16(), resp.text().await.unwrap())
        }
    };

    let busy = tokio::spawn(call());
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    let (status, body) = call().await;
    assert_eq!(status, 200);
    assert!(body.contains("from-openrouter"), "{body}");
    busy.await.unwrap();

    assert_eq!(remote.received_requests().await.unwrap().len(), 1);
    let rows = router.wait_for_logs(2).await;
    let shed = rows.iter().find(|r| r["backend"] == "openrouter").unwrap();
    let err = shed["error"].as_str().unwrap();
    // Only one shed note: the second grouped backend was skipped without
    // being probed, because its device had already answered.
    assert_eq!(err.matches("shed").count(), 1, "{err}");
}

#[tokio::test]
async fn exclusive_group_admin_crud() {
    let a = echo_backend("from-a").await;
    let router = TestRouter::start(
        &(backend_toml("b1", &a.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;

    // A backend cannot join a group that does not exist yet.
    let grouped = |group: &str| {
        serde_json::json!({
            "scheduling_class": "grouped", "exclusive_group_id": group,
            "max_concurrency": 4, "max_tokens_cap": 1024, "tok_s_estimate": 90.0
        })
    };
    let (status, err) = router
        .admin(
            reqwest::Method::PATCH,
            "/admin/backends/b1",
            Some(grouped("dgx")),
        )
        .await;
    assert_eq!(status, 400);
    assert!(
        err["error"]["message"].as_str().unwrap().contains("dgx"),
        "{err}"
    );

    let (status, created) = router
        .admin(
            reqwest::Method::POST,
            "/admin/groups",
            Some(serde_json::json!({
                "id": "dgx", "min_hold_ms": 5000, "max_hold_ms": 30000, "drain_timeout_ms": 45000
            })),
        )
        .await;
    assert_eq!(status, 201);
    assert_eq!(created["min_hold_ms"], 5000);
    assert_eq!(
        router
            .admin(
                reqwest::Method::POST,
                "/admin/groups",
                Some(serde_json::json!({
                    "id": "dgx", "min_hold_ms": 1, "max_hold_ms": 2, "drain_timeout_ms": 3
                })),
            )
            .await
            .0,
        409
    );

    // Ordering is enforced on write, not just at startup.
    let (status, _) = router
        .admin(
            reqwest::Method::PUT,
            "/admin/groups/dgx",
            Some(serde_json::json!({
                "min_hold_ms": 40000, "max_hold_ms": 30000, "drain_timeout_ms": 45000
            })),
        )
        .await;
    assert_eq!(status, 400, "min_hold_ms above max_hold_ms is rejected");

    // Now the backend can join, and the fields survive the round trip.
    let (status, view) = router
        .admin(
            reqwest::Method::PATCH,
            "/admin/backends/b1",
            Some(grouped("dgx")),
        )
        .await;
    assert_eq!(status, 200, "{view}");
    assert_eq!(view["scheduling_class"], "grouped");
    assert_eq!(view["exclusive_group_id"], "dgx");
    assert_eq!(view["max_tokens_cap"], 1024);

    // A grouped backend without a token cap would leave drain time — and
    // every peer's wait estimate — unbounded.
    let (status, _) = router
        .admin(
            reqwest::Method::PATCH,
            "/admin/backends/b1",
            Some(serde_json::json!({"max_tokens_cap": null})),
        )
        .await;
    assert_eq!(status, 400);

    // A group with members cannot be deleted.
    let (status, err) = router
        .admin(reqwest::Method::DELETE, "/admin/groups/dgx", None)
        .await;
    assert_eq!(status, 409);
    assert!(
        err["error"]["message"].as_str().unwrap().contains("b1"),
        "{err}"
    );

    // Model-level QoS and a fallback target round-trip through the API.
    let (status, _) = router
        .admin(
            reqwest::Method::POST,
            "/admin/backends",
            Some(serde_json::json!({
                "name": "remote", "kinds": ["openai"], "base_url": "https://openrouter.ai/api/v1"
            })),
        )
        .await;
    assert_eq!(status, 201);
    let (status, _) = router
        .admin(
            reqwest::Method::PUT,
            "/admin/models/gpt-test",
            Some(serde_json::json!({
                "qos_max_wait_ms": 2000,
                "qos_background_max_wait_ms": 60000,
                "on_shed_without_fallback": "error",
                "targets": [
                    {"backend": "b1"},
                    {"backend": "remote", "model": "qwen/qwen3-30b-a3b", "fallback_only": true}
                ]
            })),
        )
        .await;
    assert_eq!(status, 200);
    let (_, models) = router
        .admin(reqwest::Method::GET, "/admin/models/gpt-test", None)
        .await;
    assert_eq!(models["qos_max_wait_ms"], 2000);
    assert_eq!(models["on_shed_without_fallback"], "error");
    assert_eq!(models["targets"][1]["fallback_only"], true);

    // A route whose every target is fallback-only has nothing serving it.
    let (status, _) = router
        .admin(
            reqwest::Method::PUT,
            "/admin/models/gpt-test",
            Some(serde_json::json!({
                "targets": [{"backend": "remote", "fallback_only": true}]
            })),
        )
        .await;
    assert_eq!(status, 400);

    // Freeing the last member lets the group go.
    let (status, _) = router
        .admin(
            reqwest::Method::PATCH,
            "/admin/backends/b1",
            Some(
                serde_json::json!({"scheduling_class": "unrestricted", "exclusive_group_id": null}),
            ),
        )
        .await;
    assert_eq!(status, 200);
    assert_eq!(
        router
            .admin(reqwest::Method::DELETE, "/admin/groups/dgx", None)
            .await
            .0,
        204
    );
    assert_eq!(
        router
            .admin(reqwest::Method::DELETE, "/admin/groups/dgx", None)
            .await
            .0,
        404
    );
}

#[tokio::test]
async fn model_aliases_resolve_to_the_canonical_model() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })))
        .mount(&mock)
        .await;

    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\naliases = [\"gpt-legacy\", \"gpt4\"]\n\
               [[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    /// Status of a chat request naming `model`.
    async fn chat(router: &TestRouter, token: &str, model: &str) -> u16 {
        router
            .client
            .post(router.url("/v1/chat/completions"))
            .bearer_auth(token)
            .body(format!(r#"{{"model":"{model}","messages":[]}}"#))
            .send()
            .await
            .unwrap()
            .status()
            .as_u16()
    }

    for name in ["gpt-test", "gpt-legacy", "gpt4"] {
        assert_eq!(chat(&router, &token, name).await, 200, "{name} must route");
    }

    // A backend never hears an alias: absent a target rename, the canonical
    // name is what gets forwarded.
    let received = mock.received_requests().await.unwrap();
    assert_eq!(received.len(), 3);
    for r in &received {
        let body: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(body["model"], "gpt-test");
    }
    // ...and all three calls are logged and billed under it too.
    let logs = router.wait_for_logs(3).await;
    assert!(
        logs.iter().all(|l| l["model"] == "gpt-test"),
        "aliased traffic aggregates under the canonical name: {logs:?}"
    );

    // Allowlists reference model names; an alias entry would never match
    // (the check runs after resolution), so it is rejected outright.
    let (status, err) = router
        .admin(
            reqwest::Method::POST,
            "/admin/tokens",
            Some(serde_json::json!({"name": "aliased", "allowed_models": ["gpt-legacy"]})),
        )
        .await;
    assert_eq!(status, 400);
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("gpt-test"),
        "{err}"
    );

    // A key allowed the model may call it under any of its names.
    let (status, created) = router
        .admin(
            reqwest::Method::POST,
            "/admin/tokens",
            Some(serde_json::json!({"name": "ok", "allowed_models": ["gpt-test"]})),
        )
        .await;
    assert_eq!(status, 201);
    let restricted = created["token"].as_str().unwrap().to_string();
    assert_eq!(chat(&router, &restricted, "gpt-legacy").await, 200);

    // One inbound name, one route: neither a name nor an alias may collide
    // with a name or alias another model already answers to.
    for body in [
        serde_json::json!({"name": "gpt-legacy", "targets": [{"backend": "b1"}]}),
        serde_json::json!({"name": "new", "aliases": ["gpt4"], "targets": [{"backend": "b1"}]}),
        serde_json::json!({"name": "new", "aliases": ["dup", "dup"], "targets": [{"backend": "b1"}]}),
        serde_json::json!({"name": "new", "aliases": ["new"], "targets": [{"backend": "b1"}]}),
    ] {
        let (status, _) = router
            .admin(reqwest::Method::POST, "/admin/models", Some(body))
            .await;
        assert_eq!(status, 400);
    }

    // Aliases are editable (unlike the name): a target rename wins over the
    // canonical-name fallback, and a dropped alias stops routing.
    let (status, updated) = router
        .admin(
            reqwest::Method::PUT,
            "/admin/models/gpt-test",
            Some(serde_json::json!({
                "aliases": ["gpt-legacy"],
                "targets": [{"backend": "b1", "model": "real-upstream"}]
            })),
        )
        .await;
    assert_eq!(status, 200);
    assert_eq!(updated["aliases"], serde_json::json!(["gpt-legacy"]));
    assert_eq!(chat(&router, &token, "gpt-legacy").await, 200);
    let received = mock.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received.last().unwrap().body).unwrap();
    assert_eq!(body["model"], "real-upstream");
    assert_eq!(
        chat(&router, &token, "gpt4").await,
        404,
        "a removed alias stops routing"
    );
}

#[expect(
    unsafe_code,
    reason = "integration test sets environment variable for backend secret resolution"
)]
#[tokio::test]
async fn backend_patch_swaps_credentials_and_keeps_untouched_fields() {
    let mock = echo_backend("ok").await;
    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;

    // Absent fields stay as they are; setting an env var clears the literal.
    // SAFETY: isolated integration test sets test environment variable.
    unsafe {
        std::env::set_var("LLM_ROUTER_TEST_BACKEND_KEY", "sk-from-env");
    }
    let (status, patched) = router
        .admin(
            reqwest::Method::PATCH,
            "/admin/backends/b1",
            Some(serde_json::json!({"api_key_env": "LLM_ROUTER_TEST_BACKEND_KEY"})),
        )
        .await;
    assert_eq!(status, 200);
    assert_eq!(patched["api_key_env"], "LLM_ROUTER_TEST_BACKEND_KEY");
    assert_eq!(patched["has_api_key"], false);
    assert_eq!(patched["base_url"], mock.uri());
    assert_eq!(patched["inject_usage"], true);

    // Explicit nulls clear the credential entirely.
    let (status, patched) = router
        .admin(
            reqwest::Method::PATCH,
            "/admin/backends/b1",
            Some(serde_json::json!({"api_key": null, "api_key_env": null, "inject_usage": false})),
        )
        .await;
    assert_eq!(status, 200);
    assert!(patched["api_key_env"].is_null());
    assert_eq!(patched["has_api_key"], false);
    assert_eq!(patched["inject_usage"], false);

    assert_eq!(
        router
            .admin(
                reqwest::Method::PATCH,
                "/admin/backends/missing",
                Some(serde_json::json!({}))
            )
            .await
            .0,
        404
    );
}

#[tokio::test]
async fn catalog_persists_across_restarts() {
    let tmp = tempfile::tempdir().unwrap();
    let sqlite_path = tmp.path().join("t.db");
    let raw = format!(
        "listen = \"127.0.0.1:0\"\nsqlite_path = {:?}\nadmin_token = \"t\"\n",
        sqlite_path.to_str().unwrap()
    );
    let boot = || {
        let cfg: Config = toml::from_str(&raw).unwrap();
        cfg.validate().unwrap();
        let db = Db::open(&cfg.sqlite_path).unwrap();
        let (sink, _writer) = shiso_router::logging::start_writer(db.clone());
        AppState::from_config(&cfg, db, sink).unwrap()
    };

    let state = boot();
    assert!(state.routing().backends.is_empty());
    let b = shiso_router::catalog::BackendDef {
        name: "stored".into(),
        kinds: vec![shiso_router::catalog::BackendKind::Openai],
        base_url: "http://127.0.0.1:1".into(),
        api_key: None,
        api_key_env: None,
        anthropic_version: None,
        inject_usage: true,
        scheduling_class: shiso_router::catalog::SchedulingClass::Unrestricted,
        exclusive_group_id: None,
        max_concurrency: None,
        max_tokens_cap: None,
        tok_s_estimate: None,
    };
    shiso_router::catalog::insert_backend(&state.db, &b).unwrap();
    state.reload_routing().unwrap();
    assert_eq!(state.routing().backends[0].name, "stored");
    drop(state);

    // A second boot against the same database reloads the stored catalog.
    let state = boot();
    let routing = state.routing();
    assert_eq!(routing.backends.len(), 1);
    assert_eq!(routing.backends[0].name, "stored");
}

#[tokio::test]
async fn log_rows_and_key_usage_expose_the_input_output_split() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500}
        })))
        .mount(&mock)
        .await;
    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, id) = router.create_token().await;

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    router.wait_for_logs(1).await;

    // The key list carries the current window split, not just the total.
    let (status, keys) = router
        .admin(reqwest::Method::GET, "/admin/tokens", None)
        .await;
    assert_eq!(status, 200);
    let usage = &keys[0]["usage"];
    assert_eq!(usage["window_tokens"], 1500);
    assert_eq!(usage["window_prompt_tokens"], 1000);
    assert_eq!(usage["window_completion_tokens"], 500);

    // Log rows name the key, so the UI never has to show a raw id.
    let (status, page) = router
        .admin(reqwest::Method::GET, "/admin/logs", None)
        .await;
    assert_eq!(status, 200);
    let row = &page["items"][0];
    assert_eq!(row["token_id"], id);
    assert_eq!(row["token_name"], "test");

    let detail_path = format!("/admin/logs/{}", row["id"].as_str().unwrap());
    let (status, detail) = router.admin(reqwest::Method::GET, &detail_path, None).await;
    assert_eq!(status, 200);
    assert_eq!(detail["token_name"], "test");

    // Logs outlive their key; the name then reads as null rather than stale.
    assert_eq!(
        router
            .admin(
                reqwest::Method::DELETE,
                &format!("/admin/tokens/{id}"),
                None
            )
            .await
            .0,
        204
    );
    let (_, page) = router
        .admin(reqwest::Method::GET, "/admin/logs", None)
        .await;
    assert!(page["items"][0]["token_name"].is_null());
}

#[tokio::test]
async fn models_endpoint_lists_what_the_key_can_call() {
    let router = TestRouter::start(
        &(backend_toml("b1", "http://127.0.0.1:1", "k")
            + "[[models]]\nname = \"gpt-test\"\naliases = [\"gpt-alias\"]\n[[models.targets]]\nbackend = \"b1\"\n"
            + "[[models]]\nname = \"other-model\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    // No credential → 401; the fallback stays a plain 404.
    let resp = router
        .client
        .get(router.url("/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    // Default shape is OpenAI's: canonical names only, aliases folded in.
    let resp = router
        .client
        .get(router.url("/v1/models"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["gpt-test", "other-model"]);
    assert_eq!(body["data"][0]["object"], "model");

    // An anthropic-version header (the Anthropic SDK always sends one)
    // switches to Anthropic's list shape. x-api-key auth works like the
    // proxy endpoints. Also served prefix-less for /v1-suffixed base URLs.
    for path in ["/v1/models", "/models", "/v1/v1/models"] {
        let resp = router
            .client
            .get(router.url(path))
            .header("x-api-key", &token)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["data"][0]["type"], "model");
        assert_eq!(body["data"][0]["id"], "gpt-test");
        assert_eq!(body["first_id"], "gpt-test");
        assert_eq!(body["last_id"], "other-model");
        assert_eq!(body["has_more"], false);
    }

    // A key with an allowlist sees only its models.
    let resp = router
        .client
        .post(router.url("/admin/tokens"))
        .bearer_auth(&router.admin_token)
        .json(&serde_json::json!({"name": "restricted", "allowed_models": ["other-model"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let restricted = resp.json::<serde_json::Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = router
        .client
        .get(router.url("/v1/models"))
        .bearer_auth(&restricted)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["other-model"]);
}

// ---- the in-flight registry -------------------------------------------

impl TestRouter {
    /// Polls `/admin/active` the way the logs page does, until the set has
    /// `expected` entries. Returns the entries.
    async fn wait_for_active(&self, expected: usize) -> Vec<serde_json::Value> {
        let mut last = serde_json::Value::Null;
        for _ in 0..200 {
            let (status, page) = self
                .admin(reqwest::Method::GET, "/admin/active", None)
                .await;
            assert_eq!(status, 200);
            let items = page["items"].as_array().unwrap().clone();
            assert_eq!(page["count"].as_u64().unwrap() as usize, items.len());
            if items.len() == expected {
                return items;
            }
            last = page;
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        panic!("expected {expected} in-flight requests, last saw {last}");
    }
}

/// The active panel's whole premise: a request is visible while it runs,
/// and gone — with a finished log row in its place — once it is done.
#[tokio::test]
async fn in_flight_requests_are_listed_until_they_finish() {
    let mock = slow_backend(800).await;
    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\naliases = [\"gpt-alias\"]\n"
            + "[[models.targets]]\nbackend = \"b1\"\nmodel = \"upstream-name\"\n"),
    )
    .await;
    let (token, token_id) = router.create_token().await;

    // An idle router reports an empty set rather than 404ing or erroring.
    assert!(router.wait_for_active(0).await.is_empty());

    // Called under the *alias*: the listed model must be the canonical
    // name, matching what the eventual log row records.
    let call = tokio::spawn({
        let client = router.client.clone();
        let url = router.url("/v1/chat/completions");
        let token = token.clone();
        async move {
            let resp = client
                .post(url)
                .bearer_auth(token)
                .body(r#"{"model":"gpt-alias","messages":[{"role":"user","content":"hi"}]}"#)
                .send()
                .await
                .unwrap();
            let status = resp.status().as_u16();
            // Read the body so the tee sees end of stream, not a disconnect.
            let _ = resp.bytes().await.unwrap();
            status
        }
    });

    let items = router.wait_for_active(1).await;
    let e = &items[0];
    assert_eq!(
        e["model"], "gpt-test",
        "the alias must resolve before registration"
    );
    assert_eq!(e["token_id"], serde_json::json!(token_id));
    assert_eq!(e["token_name"], "test");
    assert_eq!(e["path"], "/v1/chat/completions");
    assert_eq!(e["backend"], "b1");
    assert_eq!(e["streamed"], false);
    // The mock delays its *head*, so the request is past admission and
    // waiting on the provider.
    assert_eq!(
        e["phase"], "upstream",
        "expected the upstream phase, got {e}"
    );
    assert!(e["elapsed_ms"].as_i64().unwrap() >= 0);
    assert!(e["created_at"].as_str().unwrap().ends_with('Z'), "{e}");
    let active_id = e["id"].as_str().unwrap().to_string();

    assert_eq!(call.await.unwrap(), 200);

    // Finished: out of the registry, and the same id turns up in the logs
    // — one request, one identity across both views.
    assert!(router.wait_for_active(0).await.is_empty());
    let rows = router.wait_for_logs(1).await;
    assert_eq!(rows[0]["id"], serde_json::json!(active_id));
    assert_eq!(rows[0]["model"], "gpt-test");
    assert_eq!(rows[0]["upstream_model"], "upstream-name");
}

/// Requests rejected before dispatch never linger: the guard drops with
/// the handler's frame, with no relay to hand it on to.
#[tokio::test]
async fn requests_rejected_before_dispatch_leave_no_entry() {
    let mock = slow_backend(0).await;
    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"
            + "[[models]]\nname = \"other-model\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;

    // A key allowed only the *other* model: rejected by the allowlist,
    // which is checked after the request has been registered.
    let (status, created) = router
        .admin(
            reqwest::Method::POST,
            "/admin/tokens",
            Some(serde_json::json!({"name": "narrow", "allowed_models": ["other-model"]})),
        )
        .await;
    assert_eq!(status, 201, "{created}");
    let token = created["token"].as_str().unwrap().to_string();

    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(CHAT_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
    assert!(router.wait_for_active(0).await.is_empty());

    // Same for an unknown model, which is rejected before registration.
    let resp = router
        .client
        .post(router.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .body(r#"{"model":"nope","messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    assert!(router.wait_for_active(0).await.is_empty());
}

/// A grouped backend holds the lease while a second request waits: the
/// waiting one must show as `queued`, which is the distinction the panel
/// exists to make (our admission control, not the provider).
#[tokio::test]
async fn a_request_waiting_on_a_device_lease_shows_as_queued() {
    let mock = slow_backend(1_500).await;
    let router = TestRouter::start(
        &("[[exclusive_groups]]\nid = \"dev0\"\nmin_hold_ms = 60000\nmax_hold_ms = 60000\n\
           drain_timeout_ms = 60000\n"
            .to_string()
            + &backend_toml("b1", &mock.uri(), "k")
            + "scheduling_class = \"grouped\"\nexclusive_group = \"dev0\"\n\
               max_tokens_cap = 64\nmax_concurrency = 1\n"
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    let call = || {
        let client = router.client.clone();
        let url = router.url("/v1/chat/completions");
        let token = token.clone();
        tokio::spawn(async move {
            let resp = client
                .post(url)
                .bearer_auth(token)
                .body(CHAT_BODY)
                .send()
                .await
                .unwrap();
            let status = resp.status().as_u16();
            let _ = resp.bytes().await.unwrap();
            status
        })
    };
    let first = call();
    // Let the first request take the single slot before the second asks.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    let second = call();

    let items = router.wait_for_active(2).await;
    let phases: Vec<&str> = items.iter().map(|e| e["phase"].as_str().unwrap()).collect();
    assert_eq!(
        phases,
        ["upstream", "queued"],
        "the holder should be upstream and the waiter queued, longest-running first: {items:?}"
    );
    // A queued request already names the backend it is waiting on — the
    // point of recording the attempt before admission.
    assert_eq!(items[1]["backend"], "b1");

    assert_eq!(first.await.unwrap(), 200);
    assert_eq!(second.await.unwrap(), 200);
    assert!(router.wait_for_active(0).await.is_empty());
}

/// The `x-service-tier` hint is both forwarded to the provider (the client
/// owns its tier) and recorded, so the UI can badge it on the in-flight
/// request and on the finished row alike.
#[tokio::test]
async fn the_service_tier_is_forwarded_and_recorded() {
    let mock = MockServer::start().await;
    // The matcher is the forwarding assertion: no header, no match, and the
    // proxied call would 502 instead of 200.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("x-service-tier", "flex"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(std::time::Duration::from_millis(600))
                .set_body_json(serde_json::json!({
                    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3}
                })),
        )
        .mount(&mock)
        .await;

    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    let call = tokio::spawn({
        let client = router.client.clone();
        let url = router.url("/v1/chat/completions");
        let token = token.clone();
        async move {
            let resp = client
                .post(url)
                .bearer_auth(token)
                // Forwarded verbatim, which is what the mock matches on;
                // the recorded value's normalization is unit-tested.
                .header("x-service-tier", "flex")
                .body(CHAT_BODY)
                .send()
                .await
                .unwrap();
            let status = resp.status().as_u16();
            let _ = resp.bytes().await.unwrap();
            status
        }
    });

    // In flight: the panel can badge it before the request finishes.
    let items = router.wait_for_active(1).await;
    assert_eq!(items[0]["service_tier"], "flex", "{items:?}");

    assert_eq!(
        call.await.unwrap(),
        200,
        "the tier header must reach the backend"
    );

    // And on the finished row, through the admin API the UI actually reads.
    router.wait_for_logs(1).await;
    let (status, page) = router
        .admin(reqwest::Method::GET, "/admin/logs", None)
        .await;
    assert_eq!(status, 200);
    let row = &page["items"][0];
    assert_eq!(row["service_tier"], "flex");
    let (_, detail) = router
        .admin(
            reqwest::Method::GET,
            &format!("/admin/logs/{}", row["id"].as_str().unwrap()),
            None,
        )
        .await;
    assert_eq!(detail["service_tier"], "flex");
}

/// Ordinary traffic leaves the column null, so a badge means something.
#[tokio::test]
async fn ordinary_and_default_tier_requests_record_no_tier() {
    let mock = echo_backend("plain").await;
    let router = TestRouter::start(
        &(backend_toml("b1", &mock.uri(), "k")
            + "[[models]]\nname = \"gpt-test\"\n[[models.targets]]\nbackend = \"b1\"\n"),
    )
    .await;
    let (token, _) = router.create_token().await;

    for tier in [None, Some("auto"), Some("default")] {
        let mut req = router
            .client
            .post(router.url("/v1/chat/completions"))
            .bearer_auth(&token)
            .body(CHAT_BODY);
        if let Some(t) = tier {
            req = req.header("x-service-tier", t);
        }
        assert_eq!(req.send().await.unwrap().status(), 200);
    }

    router.wait_for_logs(3).await;
    let (_, page) = router
        .admin(reqwest::Method::GET, "/admin/logs", None)
        .await;
    for row in page["items"].as_array().unwrap() {
        assert_eq!(
            row["service_tier"],
            serde_json::Value::Null,
            "auto/default are what saying nothing gets you: {row}"
        );
    }
}
