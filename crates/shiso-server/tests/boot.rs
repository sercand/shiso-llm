//! The whole stack, once, against a real checkpoint on a real device.
//!
//! Ignored by default and skipped with a printed note when the machine has
//! neither, because this is the one case here that cannot run on a build
//! machine. What it proves is the thing every other test in this crate stubs
//! out: that a checkpoint loads, moves onto the engine thread, serves a route,
//! and tears down.

#![cfg(feature = "metal")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use shiso_core::Backend;
use shiso_server::cli::{Cli, Command, ServeArgs};
use shiso_server::router::build_router;
use shiso_server::scheduler::spawn_engine;
use shiso_server::state::AppState;
use tower::ServiceExt as _;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// The checkpoint this test runs against, when the machine has one.
fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL").ok()?);
    path.is_file().then_some(path)
}

/// The config the checkpoint is read with.
fn config_path() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("SHISO_TEST_MODEL_OPTIQ").ok()?);
    let path = dir.join("config.json");
    path.is_file().then_some(path)
}

/// The command line this test would have been given.
fn args(model: &Path, config: &Path) -> ServeArgs {
    let cli = <Cli as clap::Parser>::try_parse_from([
        "shiso".to_owned(),
        "serve".to_owned(),
        "--model".to_owned(),
        model.display().to_string(),
        "--config".to_owned(),
        config.display().to_string(),
        "--ctx".to_owned(),
        "1024".to_owned(),
    ])
    .expect("the command line parses");
    let Command::Serve(serve) = cli.command;
    *serve
}

#[tokio::test]
#[ignore = "needs the local checkpoint and a Metal device"]
async fn a_real_checkpoint_boots_under_the_engine_thread_and_answers_health() {
    let Some(model) = model_path() else {
        skip("no local checkpoint");
        return;
    };
    let Some(config) = config_path() else {
        skip("no local config beside the checkpoint");
        return;
    };
    if !shiso_metal::metal_available() {
        skip("no Metal device");
        return;
    }

    let args = args(&model, &config);
    args.validate().expect("the command line is servable");
    let backend: Arc<dyn Backend> = shiso_metal::MetalBackend::new().expect("a Metal device");
    let parsed = shiso_core::HfConfig::from_file(&args.config_path()).expect("the config parses");
    let loaded = shiso_server::model_factory::load_model(&backend, &args.model, &parsed, &args)
        .expect("the checkpoint loads");

    let model_id = args.model_id();
    let (engine, join) = spawn_engine(backend, loaded.model, None, &model_id, args.ctx);
    let state = Arc::new(AppState::new(engine, model_id.clone(), args.ctx));

    let response = build_router(Arc::clone(&state))
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("a GET request builds"),
        )
        .await
        .expect("the router answers");
    assert_eq!(response.status(), StatusCode::OK);

    drop(state);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the model tears down cleanly");
}

/// One real request per surface, over the real checkpoint.
///
/// What the fake-model tests cannot reach: the checkpoint's own chat template
/// rendering a real conversation, its own vocabulary encoding the result, and
/// the answer coming back in the shape a client of each surface reads. The
/// assertions are about shape and accounting rather than about wording - what
/// the model says is not this test's subject, and pinning it would make every
/// checkpoint change a test failure.
#[tokio::test]
#[ignore = "needs the local checkpoint and a Metal device"]
async fn every_surface_answers_over_a_real_checkpoint() {
    let Some(model) = model_path() else {
        skip("no local checkpoint");
        return;
    };
    let Some(config) = config_path() else {
        skip("no local config beside the checkpoint");
        return;
    };
    if !shiso_metal::metal_available() {
        skip("no Metal device");
        return;
    }

    let args = args(&model, &config);
    let backend: Arc<dyn Backend> = shiso_metal::MetalBackend::new().expect("a Metal device");
    let parsed = shiso_core::HfConfig::from_file(&args.config_path()).expect("the config parses");
    let loaded = shiso_server::model_factory::load_model(&backend, &args.model, &parsed, &args)
        .expect("the checkpoint loads")
        .model;

    let vocabulary = Arc::new(
        shiso_server::tokenizer::Vocabulary::from_file(&args.tokenizer_path())
            .expect("the checkpoint's vocabulary loads"),
    );
    let template = std::fs::read_to_string(args.chat_template_path())
        .expect("the checkpoint ships a chat template");
    let renderer = shiso_server::tokenizer::render_prompt::PromptRenderer::new(
        &template,
        shiso_server::tokenizer::jinja_helpers::ToolJsonStyle::Compact,
        Arc::clone(&vocabulary),
    )
    .expect("the checkpoint's chat template compiles");

    let model_id = args.model_id();
    let (engine, join) = spawn_engine(
        backend,
        loaded,
        Some(Arc::clone(&vocabulary)),
        &model_id,
        args.ctx,
    );
    let state = Arc::new(
        AppState::new(engine, model_id, args.ctx)
            .with_vocabulary(vocabulary)
            .with_renderer(Arc::new(renderer)),
    );

    assert_surfaces_answer(&state).await;
    assert_messages_surface_answers(&state).await;

    drop(state);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the model tears down cleanly");
}

/// The four requests the surfaces are checked with.
///
/// Split out of the test body so that the test reads as setup, exercise and
/// teardown rather than as one long script.
async fn assert_surfaces_answer(state: &Arc<AppState>) {
    let chat = post_json(
        state,
        "/v1/chat/completions",
        &serde_json::json!({
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_completion_tokens": 8,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(chat["object"], "chat.completion", "{chat}");
    assert!(
        chat["choices"][0]["message"]["content"].is_string()
            || chat["choices"][0]["message"]["content"].is_null(),
        "{chat}"
    );
    assert!(
        chat["usage"]["prompt_tokens"]
            .as_u64()
            .is_some_and(|n| n > 0),
        "the accounting names the prompt that was rendered: {chat}"
    );

    let responses = post_json(
        state,
        "/v1/responses",
        &serde_json::json!({
            "input": "The capital of France is",
            "max_output_tokens": 8,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(responses["object"], "response", "{responses}");
    assert!(
        responses["output"]
            .as_array()
            .is_some_and(|items| !items.is_empty()),
        "{responses}"
    );

    let completion = post_json(
        state,
        "/v1/completions",
        &serde_json::json!({
            "prompt": "The capital of France is",
            "max_tokens": 8,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(completion["object"], "text_completion", "{completion}");

    let stream = post_text(
        state,
        "/v1/chat/completions",
        &serde_json::json!({
            "messages": [{"role": "user", "content": "Count to three."}],
            "max_completion_tokens": 8,
            "temperature": 0.0,
            "stream": true
        }),
    )
    .await;
    assert!(stream.contains("data: [DONE]"), "{stream}");
    for frame in stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|frame| *frame != "[DONE]")
    {
        let body: serde_json::Value =
            serde_json::from_str(frame).expect("every streamed frame is JSON");
        assert_eq!(body["object"], "chat.completion.chunk", "{body}");
    }
}

/// The two requests the Messages surface is checked with.
///
/// It shares the loaded checkpoint with the other surfaces rather than booting
/// a second one: loading is the slow part, and two loads would make the one
/// test that cannot run on a build machine twice as expensive for no extra
/// coverage.
async fn assert_messages_surface_answers(state: &Arc<AppState>) {
    let message = post_json(
        state,
        "/v1/messages",
        &serde_json::json!({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 8,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(message["type"], "message", "{message}");
    assert_eq!(message["role"], "assistant", "{message}");
    assert!(
        message["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("msg_")),
        "{message}"
    );
    // What the model says is not this test's subject, so the assertion is that
    // the turn ended one of the two ways a short budget allows.
    assert!(
        matches!(
            message["stop_reason"].as_str(),
            Some("end_turn" | "max_tokens")
        ),
        "{message}"
    );
    assert!(
        message["usage"]["input_tokens"]
            .as_u64()
            .is_some_and(|n| n > 0),
        "the accounting names the prompt that was rendered: {message}"
    );

    let stream = post_text(
        state,
        "/v1/messages",
        &serde_json::json!({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "Count to three."}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": true
        }),
    )
    .await;
    let names: Vec<&str> = stream
        .lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .collect();
    assert_eq!(names.first(), Some(&"message_start"), "{stream}");
    assert_eq!(names.last(), Some(&"message_stop"), "{stream}");
    assert!(
        names.contains(&"content_block_start") && names.contains(&"message_delta"),
        "{stream}"
    );
    for frame in stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
    {
        let body: serde_json::Value =
            serde_json::from_str(frame).expect("every streamed frame is JSON");
        assert!(body["type"].is_string(), "{body}");
    }
}

/// POSTs a body and reads the answer as JSON.
async fn post_json(
    state: &Arc<AppState>,
    path: &str,
    body: &serde_json::Value,
) -> serde_json::Value {
    let text = post_text(state, path, body).await;
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("{path} answered {text}: {error}"))
}

/// POSTs a body and reads the answer as text.
async fn post_text(state: &Arc<AppState>, path: &str, body: &serde_json::Value) -> String {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("a POST request builds");
    let response = build_router(Arc::clone(state))
        .oneshot(request)
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let text = String::from_utf8(bytes.to_vec()).expect("the body is text");
    // The body is in the message because a refusal that says only "400" is a
    // failure nobody can diagnose from the test output.
    assert_eq!(status, StatusCode::OK, "{path} answered {text}");
    text
}

/// A running server with its checkpoint tier open, and what a test needs to
/// watch it.
struct CheckpointedServer {
    /// The state its handlers are bound to.
    serving: Arc<AppState>,
    /// The engine thread, to be joined after the state is dropped.
    join: std::thread::JoinHandle<Result<(), shiso_server::scheduler::EngineFailure>>,
    /// The coordinator's counters.
    counters: Arc<shiso_server::scheduler::SchedulerStats>,
    /// The floor the store refuses shorter prefixes under.
    min_tokens: u32,
}

/// A server whose checkpoint tier is open over `dir`, and the counters behind
/// it.
///
/// Split out of the test below so that the test body reads as exercise and
/// assertion: the setup is six things that have to happen in one order - the
/// tier before any session, the renderer before the state - and none of them is
/// what the test is about.
fn checkpointed_server(args: &ServeArgs, dir: &Path) -> CheckpointedServer {
    let backend: Arc<dyn Backend> = shiso_metal::MetalBackend::new().expect("a Metal device");
    let parsed = shiso_core::HfConfig::from_file(&args.config_path()).expect("the config parses");
    let mut loaded = shiso_server::model_factory::load_model(&backend, &args.model, &parsed, args)
        .expect("the checkpoint loads")
        .model;

    // Before any session exists: the tier is a property of the cache the
    // sessions draw their pages from.
    let mut options = shiso_engine::ckpt::CkptOptions::new();
    options.budget_mb = args.kvstore_budget_mb;
    let ckpt = shiso_engine::ckpt::CkptConfig::new(dir).with_options(options);
    shiso_engine::ckpt::Checkpointed::kv_enable(loaded.as_mut(), &ckpt)
        .expect("the checkpoint tier opens over an empty directory");

    let vocabulary = Arc::new(
        shiso_server::tokenizer::Vocabulary::from_file(&args.tokenizer_path())
            .expect("the checkpoint's vocabulary loads"),
    );
    let template = std::fs::read_to_string(args.chat_template_path())
        .expect("the checkpoint ships a chat template");
    let renderer = shiso_server::tokenizer::render_prompt::PromptRenderer::new(
        &template,
        shiso_server::tokenizer::jinja_helpers::ToolJsonStyle::Compact,
        Arc::clone(&vocabulary),
    )
    .expect("the checkpoint's chat template compiles");

    let model_id = args.model_id();
    let (engine, join) = shiso_server::scheduler::spawn_engine_with_checkpoints(
        backend,
        loaded,
        None,
        Some(Arc::clone(&vocabulary)),
        &model_id,
        args.ctx,
        shiso_server::scheduler::SchedulerConfig::new(),
        (
            shiso_parsers::ReasoningScheme::None,
            shiso_parsers::ToolCallScheme::None,
        ),
    );
    let counters = Arc::clone(engine.stats());
    let serving = Arc::new(
        AppState::new(engine, model_id, args.ctx)
            .with_vocabulary(vocabulary)
            .with_renderer(Arc::new(renderer))
            .with_checkpoint_tier(shiso_server::state::CheckpointTier {
                dir: dir.to_path_buf(),
                budget_bytes: args
                    .kvstore_budget_mb
                    .unwrap_or(0)
                    .saturating_mul(1024 * 1024),
                min_tokens: options.min_tokens,
            }),
    );
    CheckpointedServer {
        serving,
        join,
        counters,
        min_tokens: options.min_tokens,
    }
}

/// The disk tier, opened for real and exercised twice.
///
/// This is the only place the checkpoint path runs end to end: a fake tier
/// cannot hand back a resume, because the type that carries one has no
/// constructor outside the engine. So the proof that `--kvstore-dir` does
/// something is here - a prompt long enough to clear the store's own floor is
/// served once cold, written, and then served again from what was written.
#[tokio::test]
#[ignore = "needs the local checkpoint and a Metal device"]
async fn a_prompt_served_twice_is_prefilled_once_and_resumed_once() {
    let Some(model) = model_path() else {
        skip("no local checkpoint");
        return;
    };
    let Some(config) = config_path() else {
        skip("no local config beside the checkpoint");
        return;
    };
    if !shiso_metal::metal_available() {
        skip("no Metal device");
        return;
    }

    let dir = tempfile::tempdir().expect("a temporary cache directory");
    let mut args = args(&model, &config);
    args.kvstore_dir = Some(dir.path().to_path_buf());
    let CheckpointedServer {
        serving,
        join,
        counters,
        min_tokens,
    } = checkpointed_server(&args, dir.path());

    // Long enough to clear the store's own floor and short enough to leave room
    // to answer; a shorter prompt is refused by the engine and this test would
    // pass without ever writing a file.
    let long = "the capital of france is paris and ".repeat(100);
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": long}],
        "max_completion_tokens": 1,
        "temperature": 0.0
    });

    let first = post_json(&serving, "/v1/chat/completions", &body).await;
    assert_eq!(first["object"], "chat.completion", "{first}");
    assert_eq!(
        counters.checkpoint_hits(),
        0,
        "the first run has nothing to resume from"
    );
    assert!(
        counters.checkpoint_writes() > 0,
        "a prompt over the {min_tokens}-token floor is written: {} writes",
        counters.checkpoint_writes()
    );
    let prefilled_once = counters.prefill_tokens();
    assert!(prefilled_once > 0);

    let second = post_json(&serving, "/v1/chat/completions", &body).await;
    assert_eq!(second["object"], "chat.completion", "{second}");
    assert_eq!(
        counters.checkpoint_hits(),
        1,
        "the same rendered prompt resumes from what the first run wrote"
    );
    assert!(
        counters.reused_tokens() > 0,
        "the resume covered prompt tokens rather than reporting a hit of nothing"
    );
    assert!(
        counters.prefill_tokens() - prefilled_once < prefilled_once,
        "the second run prefilled less than the first: {prefilled_once} then {}",
        counters.prefill_tokens() - prefilled_once
    );
    assert_eq!(
        second["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or_default(),
        counters.reused_tokens(),
        "what was reused is what the caller is told was cached: {second}"
    );

    // And the operator endpoint says the tier is on rather than disabled.
    let report = shiso_server::api::misc_handlers::cache_body(&serving);
    assert!(report.disk.enabled, "{report:?}");
    assert_eq!(report.engine.hits, 1);

    drop(serving);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the model tears down cleanly");
}
