//! The typed core against the `/v1/decisions` route, on a real d1 checkpoint
//! and a real device.
//!
//! The route parses the wire into `shiso_ir::decision`, calls
//! `shiso_runtime::decision::decide`, and spells the answer back out. This is
//! the test that the wire layer in between changes nothing: the same request
//! handed to `decide` as IR answers what the endpoint answers, to within
//! arithmetic noise.
//!
//! ```text
//! SHISO_HOME=/path/to/home cargo test --release -p shiso-server --features metal \
//!   --test decisions_in_process -- --ignored --nocapture
//! ```
//!
//! The model is `d1-3b` installed under `$SHISO_HOME/models/d1-3b/q8_0`, or
//! the directory `SHISO_TEST_MODEL_D1` names.

#![cfg(feature = "metal")]

mod decision_cases;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::Value;
use shiso_core::Backend;
use shiso_runtime::decision::decide;
use shiso_server::cli::{Cli, Command};
use shiso_server::router::build_router;
use shiso_server::scheduler::{SchedulerConfig, spawn_engine_with_checkpoints};
use shiso_server::state::AppState;
use tower::ServiceExt as _;

use decision_cases::{assert_same_answers, cases, request, wire_body};

/// Says why a case skipped, on stderr.
#[expect(clippy::print_stderr, reason = "a skipped case says why on stderr")]
fn say(line: &str) {
    eprintln!("{line}");
}

/// The installed d1 directory: `config/` beside `base/`.
fn d1_dir() -> Option<PathBuf> {
    let dir = match std::env::var_os("SHISO_TEST_MODEL_D1") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(std::env::var_os("SHISO_HOME")?).join("models/d1-3b/q8_0"),
    };
    dir.join("config/config.json").is_file().then_some(dir)
}

/// The checkpoint under `base/`, leaving the vision sidecar out.
fn checkpoint(dir: &Path) -> PathBuf {
    std::fs::read_dir(dir.join("base"))
        .expect("the install has a base directory")
        .map(|entry| entry.expect("an entry").path())
        .find(|path| {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            path.extension().is_some_and(|ext| ext == "gguf") && !name.starts_with("mmproj")
        })
        .expect("the install holds a checkpoint")
}

/// A booted server over the checkpoint, and its engine thread.
struct Stack {
    state: Arc<AppState>,
    engine: std::thread::JoinHandle<Result<(), shiso_server::scheduler::EngineFailure>>,
}

impl Stack {
    /// Boots the way the binary does, stopped short of a socket.
    fn boot(dir: &Path) -> Self {
        let model = checkpoint(dir).display().to_string();
        let config = dir.join("config/config.json").display().to_string();
        let line = [
            "shiso", "serve", "--model", &model, "--config", &config, "--ctx", "2048",
        ];
        let cli = <Cli as clap::Parser>::try_parse_from(line).expect("the command line parses");
        let Command::Serve(args) = cli.command else {
            panic!("expected Command::Serve");
        };
        args.validate().expect("the command line is servable");
        let backend: Arc<dyn Backend> = shiso_metal::MetalBackend::new().expect("a Metal device");
        let parsed = shiso_core::HfConfig::from_file(&args.config_path()).expect("the config");
        let shiso_server::model_factory::Loaded {
            model: loaded,
            defaults,
            ..
        } = shiso_server::model_factory::load_model(
            &backend,
            args.model.as_deref().expect("--model was passed"),
            &parsed,
            &args,
        )
        .expect("the checkpoint loads");
        let decision_readout = loaded.decision_readout();
        let vocabulary = Arc::new(
            shiso_server::tokenizer::Vocabulary::from_file(&args.tokenizer_path())
                .expect("the vocabulary loads"),
        );
        let template = std::fs::read_to_string(args.chat_template_path())
            .expect("the checkpoint ships a chat template");
        let renderer = shiso_server::tokenizer::render_prompt::PromptRenderer::new(
            &template,
            shiso_server::tokenizer::jinja_helpers::ToolJsonStyle::Compact,
            Arc::clone(&vocabulary),
        )
        .expect("the chat template compiles")
        .with_thinking_default(defaults.thinking_default);
        let model_id = args.model_id();
        let (engine, join) = spawn_engine_with_checkpoints(
            backend,
            loaded,
            shiso_server::media::MediaTowers::default(),
            Some(Arc::clone(&vocabulary)),
            &model_id,
            args.ctx,
            SchedulerConfig::new(),
            (defaults.reasoning, defaults.tools),
        );
        let mut state = AppState::new(engine, model_id, args.ctx)
            .with_vocabulary(vocabulary)
            .with_renderer(Arc::new(renderer));
        state.decision_readout = decision_readout;
        Self {
            state: Arc::new(state),
            engine: join,
        }
    }

    fn shutdown(self) {
        drop(self.state);
        self.engine
            .join()
            .expect("the engine thread does not panic")
            .expect("the model tears down cleanly");
    }
}

/// One POST to `/v1/decisions`, in process.
async fn post_decisions(state: &Arc<AppState>, body: &Value) -> Value {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/decisions")
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
    let body: Value = serde_json::from_slice(&bytes).expect("the answer is JSON");
    assert_eq!(status, StatusCode::OK, "{body}");
    body
}

#[test]
#[ignore = "needs d1-3b under SHISO_HOME (or SHISO_TEST_MODEL_D1) and a Metal device"]
fn d1_answers_match_the_endpoint() {
    let Some(dir) = d1_dir() else {
        say(
            "skipping: neither SHISO_TEST_MODEL_D1 nor SHISO_HOME/models/d1-3b/q8_0 names an installed d1",
        );
        return;
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime");
    let stack = Stack::boot(&dir);

    for (input, questions) in cases() {
        let served = runtime.block_on(post_decisions(&stack.state, &wire_body(&input, &questions)));
        let local = runtime
            .block_on(decide(
                &stack.state,
                request(
                    &input,
                    questions
                        .into_iter()
                        .map(|(question, _)| question)
                        .collect(),
                ),
            ))
            .expect("decide answers")
            .answers;
        assert_same_answers(&served, &local, &input);
    }
    stack.shutdown();
}
