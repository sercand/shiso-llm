//! End-to-end parsers verification with a live Gemma 4 checkpoint.
//!
//! Verifies that thinking blocks and tool calls are separated into their own
//! structures on the wire for Gemma 4 output, and that `--reasoning-parser none`
//! restores the raw delimited stream.

#![cfg(feature = "metal")]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use shiso_core::Backend;
use shiso_server::cli::{Cli, Command, ServeArgs};
use shiso_server::router::build_router;
use shiso_server::scheduler::{SchedulerConfig, spawn_engine_with_checkpoints};
use shiso_server::state::AppState;

fn gemma_model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_GEMMA4_MODEL").ok()?);
    path.is_file().then_some(path)
}

fn gemma_config_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_GEMMA4_DIR").ok()?).join("config.json");
    path.is_file().then_some(path)
}

fn args(model: &Path, config: &Path, extra: &[&str]) -> ServeArgs {
    let (model, config) = (model.display().to_string(), config.display().to_string());
    let mut line = vec![
        "shiso", "serve", "--model", &model, "--config", &config, "--ctx", "2048",
    ];
    line.extend(extra);
    let cli = <Cli as clap::Parser>::try_parse_from(line).expect("the command line parses");
    let Command::Serve(serve) = cli.command;
    *serve
}

struct Stack {
    addr: SocketAddr,
    state: Arc<AppState>,
    server: tokio::task::JoinHandle<()>,
    engine: std::thread::JoinHandle<Result<(), shiso_server::scheduler::EngineFailure>>,
}

impl Stack {
    async fn boot(model: &Path, config: &Path, extra: &[&str]) -> Self {
        let args = args(model, config, extra);
        args.validate().expect("the command line is servable");
        let backend: Arc<dyn Backend> = shiso_metal::MetalBackend::new().expect("a Metal device");
        let parsed = shiso_core::HfConfig::from_file(&args.config_path()).expect("the config");
        let loaded = shiso_server::model_factory::load_model(&backend, &args.model, &parsed, &args)
            .expect("the checkpoint loads");
        let model = loaded.model;
        let defaults = loaded.defaults;

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

        let reasoning_scheme = args.reasoning_parser.reasoning(defaults.reasoning);
        let tool_call_scheme = args.tool_call_parser.tools(defaults.tools);

        let model_id = args.model_id();
        let (engine, join) = spawn_engine_with_checkpoints(
            backend,
            model,
            None,
            Some(Arc::clone(&vocabulary)),
            &model_id,
            args.ctx,
            SchedulerConfig::new(),
            (reasoning_scheme, tool_call_scheme),
        );
        let state = Arc::new(
            AppState::new(engine, model_id, args.ctx)
                .with_vocabulary(vocabulary)
                .with_renderer(Arc::new(renderer)),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binds");
        let addr = listener.local_addr().expect("the bound port reads back");
        let router =
            build_router(Arc::clone(&state)).into_make_service_with_connect_info::<SocketAddr>();
        let server = tokio::spawn(async move {
            let _served = axum::serve(listener, router).await;
        });
        Self {
            addr,
            state,
            server,
            engine: join,
        }
    }

    async fn shutdown(self) {
        self.server.abort();
        let _stopped = self.server.await;
        drop(self.state);
        self.engine
            .join()
            .expect("the engine thread does not panic")
            .expect("the model tears down cleanly");
    }
}

fn req(addr: SocketAddr, method: &str, path: &str) -> axum::http::request::Builder {
    Request::builder()
        .method(method)
        .uri(path)
        .header("host", addr.to_string())
}

async fn send(addr: SocketAddr, request: Request<Body>) -> hyper::Response<hyper::body::Incoming> {
    let socket = tokio::net::TcpStream::connect(addr)
        .await
        .expect("it accepts");
    let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(socket))
        .await
        .expect("the server speaks HTTP/1.1");
    tokio::spawn(async move {
        let _ended = conn.await;
    });
    sender
        .send_request(request)
        .await
        .expect("the server answers")
}

async fn open(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> hyper::Response<hyper::body::Incoming> {
    let builder = req(addr, method, path);
    let request = match body {
        Some(value) => builder
            .header("content-type", "application/json")
            .body(Body::from(value.to_string())),
        None => builder.body(Body::empty()),
    };
    send(addr, request.expect("the request builds")).await
}

async fn read(response: hyper::Response<hyper::body::Incoming>) -> (StatusCode, HeaderMap, String) {
    let (status, headers) = (response.status(), response.headers().clone());
    let bytes = axum::body::to_bytes(Body::new(response.into_body()), usize::MAX)
        .await
        .expect("the body reads");
    (
        status,
        headers,
        String::from_utf8(bytes.to_vec()).expect("the body is text"),
    )
}

async fn call(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> (StatusCode, HeaderMap, String) {
    read(open(addr, method, path, body).await).await
}

async fn post_json(addr: SocketAddr, path: &str, body: &Value) -> Value {
    let (status, _headers, text) = call(addr, "POST", path, Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{path} answered {text}");
    serde_json::from_str(&text).expect("a JSON answer")
}

#[expect(clippy::print_stdout, reason = "test report is printed")]
fn note(msg: &str) {
    println!("{msg}");
}

#[tokio::test]
#[ignore = "needs the local Gemma 4 checkpoint and a Metal device"]
async fn gemma4_parsers_e2e() {
    let (Some(model), Some(config)) = (gemma_model_path(), gemma_config_path()) else {
        note("skipping: Gemma 4 checkpoint not found");
        return;
    };

    // Case (a): a request with tools declaring get_weather -> finish_reason == "tool_calls"
    // and Case (b): enable_thinking: true -> reasoning_content non-empty
    let stack = Stack::boot(&model, &config, &[]).await;

    let tools_request = json!({
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        },
                        "required": ["city"]
                    }
                }
            }
        ],
        "temperature": 0.0,
        "max_tokens": 128
    });

    let ans_a = post_json(stack.addr, "/v1/chat/completions", &tools_request).await;
    let choice_a = &ans_a["choices"][0];
    let finish_reason_a = choice_a["finish_reason"].as_str().unwrap_or_default();
    assert_eq!(finish_reason_a, "tool_calls", "full answer: {ans_a:?}");
    let tool_calls = &choice_a["message"]["tool_calls"];
    assert!(
        !tool_calls
            .as_array()
            .expect("array of tool calls")
            .is_empty()
    );
    assert_eq!(tool_calls[0]["function"]["name"], "get_weather");
    let args_str = tool_calls[0]["function"]["arguments"]
        .as_str()
        .expect("arguments string");
    let parsed_args: Value = serde_json::from_str(args_str).expect("arguments must be valid json");
    assert!(
        parsed_args.get("city").is_some(),
        "parsed args must have city: {parsed_args:?}"
    );

    // Case (b): enable_thinking: true -> reasoning_content non-empty and content non-empty
    let think_request = json!({
        "messages": [
            {
                "role": "user",
                "content": "Which number is larger: 9.11 or 9.9? Think step by step.",
            }
        ],
        "enable_thinking": true,
        "reasoning_effort": "high",
        "chat_template_kwargs": {"enable_thinking": true},
        "temperature": 0.0,
        "max_tokens": 256,
    });

    let ans_b = post_json(stack.addr, "/v1/chat/completions", &think_request).await;
    let message_b = &ans_b["choices"][0]["message"];
    let reasoning_b = message_b["reasoning_content"].as_str().unwrap_or_default();
    assert!(
        !reasoning_b.is_empty(),
        "reasoning_content must not be empty: {ans_b:?}"
    );

    stack.shutdown().await;

    // Case (c): same server started with --reasoning-parser none -> raw <|channel>thought in content
    let stack_no_reasoning = Stack::boot(&model, &config, &["--reasoning-parser", "none"]).await;
    let raw_request = json!({
        "messages": [
            {
                "role": "user",
                "content": "Which number is larger: 9.11 or 9.9? Think step by step.",
            }
        ],
        "enable_thinking": true,
        "reasoning_effort": "high",
        "chat_template_kwargs": {"enable_thinking": true},
        "temperature": 0.0,
        "max_tokens": 256,
    });

    let ans_c = post_json(
        stack_no_reasoning.addr,
        "/v1/chat/completions",
        &raw_request,
    )
    .await;
    let content_c = ans_c["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        content_c.contains("<|channel>thought"),
        "raw text must contain <|channel>thought: {content_c:?}"
    );

    stack_no_reasoning.shutdown().await;
}
