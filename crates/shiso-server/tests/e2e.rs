//! One boot of the whole stack, driven over a real loopback socket.
//!
//! The per-surface tests call a router in process, one request at a time,
//! against a model that is not there. This starts the coordinator, the engine
//! thread and the router together, binds port zero and reads the port back,
//! then speaks HTTP/1.1 to it: four surfaces blocking and streamed in one
//! session, the routes that answer without generating, and two streams open at
//! once. A socket is the only place chunked transfer, header framing and the
//! coordinator's interleaving are all real at the same time.
//!
//! Every key, prefix and vocabulary word asserted below is transcribed from the
//! published wire contracts of the two APIs this server speaks, not read back
//! out of this crate's own types. A test that serializes this server's structs
//! and checks that they deserialize is a test of `serde`: it passes just as
//! happily when every field name is wrong.

#![cfg(feature = "metal")]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use futures::{Stream, StreamExt as _};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use shiso_core::Backend;
use shiso_server::cli::{Cli, Command, ServeArgs};
use shiso_server::router::build_router;
use shiso_server::scheduler::{SchedulerConfig, spawn_engine_with_checkpoints};
use shiso_server::state::AppState;

// --- the published contracts, transcribed by hand -------------------------

/// Chat Completions: the identifier prefix, the two `object` discriminants,
/// every word `finish_reason` may take, and the counters `usage` carries.
const CHAT_ID: &str = "chatcmpl-";
const CHAT_OBJECT: &str = "chat.completion";
const CHAT_CHUNK: &str = "chat.completion.chunk";
const FINISH: [&str; 5] = [
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
];
const OPENAI_USAGE: [&str; 3] = ["prompt_tokens", "completion_tokens", "total_tokens"];

/// Completions, Responses, and the model listing.
const TEXT_COMPLETION: &str = "text_completion";
const RESPONSE_OBJECT: &str = "response";
const RESPONSE_ID: &str = "resp_";
const CREATED: &str = "response.created";
const COMPLETED: &str = "response.completed";
const LIST_OBJECT: &str = "list";
const MODEL_OBJECT: &str = "model";

/// Messages: the turn, its vocabulary, its counters, its stream's bookends.
const MESSAGE: &str = "message";
const ASSISTANT: &str = "assistant";
const STOP: [&str; 4] = ["end_turn", "max_tokens", "stop_sequence", "tool_use"];
const ANTHROPIC_USAGE: [&str; 3] = ["input_tokens", "output_tokens", "cache_read_input_tokens"];
const MESSAGE_START: &str = "message_start";
const BLOCK_DELTA: &str = "content_block_delta";
const MESSAGE_STOP: &str = "message_stop";

/// The two error envelopes, and the framing of a server-sent event.
const ERROR: &str = "error";
const INVALID: &str = "invalid_request_error";
const DATA: &str = "data: ";
const EVENT: &str = "event: ";
const DONE: &str = "[DONE]";

/// What one request over the socket yields.
type Answer = (StatusCode, HeaderMap, String);

/// A response whose body has not been read yet.
type Head = hyper::Response<hyper::body::Incoming>;

// --- what the machine has to provide --------------------------------------

/// Prints a line of the test's own report.
#[expect(clippy::print_stdout, reason = "a test's report is its output")]
fn note(what: &str) {
    println!("{what}");
}

/// The checkpoint this runs against, when the machine has one.
fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL").ok()?);
    path.is_file().then_some(path)
}

/// The architecture config the checkpoint is read with.
fn config_path() -> Option<PathBuf> {
    let var = std::env::var("SHISO_TEST_MODEL_OPTIQ");
    let path = PathBuf::from(var.ok()?).join("config.json");
    path.is_file().then_some(path)
}

/// The command line this test would have been given.
fn args(model: &Path, config: &Path) -> ServeArgs {
    let (model, config) = (model.display().to_string(), config.display().to_string());
    let line = [
        "shiso", "serve", "--model", &model, "--config", &config, "--ctx", "2048",
    ];
    let cli = <Cli as clap::Parser>::try_parse_from(line).expect("the command line parses");
    let Command::Serve(serve) = cli.command;
    *serve
}

/// A running server and the two things that must be torn down, in order.
struct Stack {
    /// What a client dials.
    addr: SocketAddr,
    /// The state the router holds; the accept task owns the other reference.
    state: Arc<AppState>,
    /// The accept task.
    server: tokio::task::JoinHandle<()>,
    /// The engine thread, joined once every queue reference is gone.
    engine: std::thread::JoinHandle<Result<(), shiso_server::scheduler::EngineFailure>>,
}

impl Stack {
    /// Loads the checkpoint, starts the coordinator, and serves on a port the
    /// operating system picks.
    async fn boot(model: &Path, config: &Path) -> Self {
        let args = args(model, config);
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

        let model_id = args.model_id();
        let (engine, join) = spawn_engine_with_checkpoints(
            backend,
            model,
            None,
            Some(Arc::clone(&vocabulary)),
            &model_id,
            args.ctx,
            SchedulerConfig::new(),
            (defaults.reasoning, defaults.tools),
        );
        let state = Arc::new(
            AppState::new(engine, model_id, args.ctx)
                .with_vocabulary(vocabulary)
                .with_renderer(Arc::new(renderer)),
        );

        // Port zero and then read it back: a fixed port makes this test fail on
        // a machine that happens to be running something else.
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

    /// Stops accepting, releases the queue, and joins the engine thread.
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

// --- the client -----------------------------------------------------------

/// A builder aimed at one route of the running server.
fn req(addr: SocketAddr, method: &str, path: &str) -> axum::http::request::Builder {
    Request::builder()
        .method(method)
        .uri(path)
        .header("host", addr.to_string())
}

/// Sends one prepared request on a connection of its own.
async fn send(addr: SocketAddr, request: Request<Body>) -> Head {
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

/// Sends a request with an optional JSON body and returns the response head.
async fn open(addr: SocketAddr, method: &str, path: &str, body: Option<&Value>) -> Head {
    let builder = req(addr, method, path);
    let request = match body {
        Some(value) => builder
            .header("content-type", "application/json")
            .body(Body::from(value.to_string())),
        None => builder.body(Body::empty()),
    };
    send(addr, request.expect("the request builds")).await
}

/// Reads a whole response.
async fn read(response: Head) -> Answer {
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

/// Sends one request and reads the whole answer.
async fn call(addr: SocketAddr, method: &str, path: &str, body: Option<&Value>) -> Answer {
    read(open(addr, method, path, body).await).await
}

/// POSTs a body, insists on a 200, and parses the answer as JSON.
async fn post_json(addr: SocketAddr, path: &str, body: &Value) -> Value {
    let (status, _headers, text) = call(addr, "POST", path, Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{path} answered {text}");
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("{path}: {text}: {error}"))
}

/// POSTs a body, insists on a 200 event stream, and returns the raw bytes.
async fn post_stream(addr: SocketAddr, path: &str, body: &Value) -> String {
    let (status, headers, text) = call(addr, "POST", path, Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{path} answered {text}");
    let kind = headers.get("content-type").and_then(|v| v.to_str().ok());
    assert!(
        kind.is_some_and(|k| k.starts_with("text/event-stream")),
        "{path}: {kind:?}"
    );
    text
}

/// Every frame of a stream: its `event:` name, when it has one, and its payload.
///
/// Fields are gathered until the blank line that dispatches them, which is what
/// the event-stream format says an event is - not a fixed pair of lines in a
/// fixed order. That matters here: this server writes `data:` before `event:`,
/// the reverse of the order both vendors print in their documentation, and a
/// parser that assumed the documented order would read every name as missing.
/// Everything else is asserted as it goes: a line is a comment, a name, a
/// payload, or the stream is not one a client can read.
fn frames(text: &str, label: &str) -> Vec<(Option<String>, String)> {
    let mut out = Vec::new();
    let (mut name, mut data) = (None, None);
    for line in text.lines() {
        if line.is_empty() {
            if let Some(payload) = data.take() {
                out.push((name, payload));
            }
            name = None;
        } else if let Some(rest) = line.strip_prefix(EVENT) {
            name = Some(rest.to_owned());
        } else if let Some(rest) = line.strip_prefix(DATA) {
            data = Some(rest.to_owned());
        } else {
            assert!(line.starts_with(':'), "{label}: unframed line {line:?}");
        }
    }
    assert!(
        data.is_none(),
        "{label}: a frame with no blank line to dispatch it"
    );
    out
}

/// The chunks of an unnamed stream, with its terminator checked.
///
/// Unnamed is itself part of the contract: this family sends only `data:`
/// lines, and a client would read a stray `event:` as a type it cannot handle.
fn unnamed(text: &str, label: &str) -> Vec<Value> {
    let all = frames(text, label);
    let (name, last) = all.last().expect("at least one frame");
    assert!(name.is_none() && last == DONE, "{label}: no {DONE}: {text}");
    all.iter()
        .take(all.len() - 1)
        .map(|(name, data)| {
            assert!(name.is_none(), "{label}: a named frame {name:?}");
            serde_json::from_str(data).expect("every chunk is JSON")
        })
        .collect()
}

/// The events of a named stream, each checked against the `type` it carries.
///
/// The `event:` line and the payload's `type` are two copies of one fact and a
/// client may read either, so they are checked against each other here.
fn named(text: &str, label: &str) -> Vec<(String, Value)> {
    frames(text, label)
        .into_iter()
        .map(|(name, data)| {
            let value: Value = serde_json::from_str(&data).expect("every event is JSON");
            let kind = value["type"].as_str().unwrap_or_default().to_owned();
            assert_eq!(name.as_deref(), Some(kind.as_str()), "{label}: {data}");
            (kind, value)
        })
        .collect()
}

/// The three counters are present, integral, and not negative.
fn counters(usage: &Value, fields: [&str; 3], label: &str) {
    for field in fields {
        let count = usage.get(field).and_then(Value::as_i64);
        assert!(count.is_some_and(|n| n >= 0), "{label}: {field} in {usage}");
    }
}

/// The value is one of the words the contract allows.
fn vocabulary(value: &Value, allowed: &[&str], label: &str) {
    let word = value.as_str().unwrap_or_default();
    assert!(
        allowed.contains(&word),
        "{label}: {value} not in {allowed:?}"
    );
}

// --- the cases ------------------------------------------------------------

/// Chat Completions, blocking and streamed, with the two compared.
async fn chat(addr: SocketAddr) {
    let mut ask = json!({
        "messages": [{"role": "user", "content": "Name the capital city of France."}],
        "max_completion_tokens": 24, "temperature": 0.0
    });
    let body = post_json(addr, "/v1/chat/completions", &ask).await;
    let id = body["id"].as_str().unwrap_or_default();
    assert!(id.starts_with(CHAT_ID), "chat: {body}");
    assert_eq!(body["object"], CHAT_OBJECT, "chat: {body}");
    let message = &body["choices"][0]["message"];
    assert_eq!(message["role"], ASSISTANT, "chat: {body}");
    let answer = message["content"].as_str().unwrap_or_default().to_owned();
    assert!(!answer.is_empty(), "chat: an empty answer in {body}");
    vocabulary(&body["choices"][0]["finish_reason"], &FINISH, "chat");
    counters(&body["usage"], OPENAI_USAGE, "chat");
    note("ok: /v1/chat/completions blocking");

    ask["stream"] = json!(true);
    let text = post_stream(addr, "/v1/chat/completions", &ask).await;
    let mut streamed = String::new();
    for chunk in unnamed(&text, "chat stream") {
        assert_eq!(chunk["object"], CHAT_CHUNK, "chat stream: {chunk}");
        let id = chunk["id"].as_str().unwrap_or_default();
        assert!(id.starts_with(CHAT_ID), "chat stream: {chunk}");
        let delta = chunk["choices"][0]["delta"]["content"].as_str();
        streamed.push_str(delta.unwrap_or_default());
    }
    // Greedy decoding is deterministic, so two deliveries of the same turn are
    // the same text. A difference is the streamed encoder dropping or repeating
    // a fragment, which no per-frame assertion can see.
    assert_eq!(streamed, answer, "chat stream: {text}");
    note("ok: /v1/chat/completions streaming, deltas equal the blocking answer");
}

/// The Responses surface, blocking and streamed.
async fn responses(addr: SocketAddr) {
    let mut ask = json!({
        "input": "Name the capital city of France.",
        "max_output_tokens": 24, "temperature": 0.0
    });
    let body = post_json(addr, "/v1/responses", &ask).await;
    assert_eq!(body["object"], RESPONSE_OBJECT, "responses: {body}");
    let id = body["id"].as_str().unwrap_or_default();
    assert!(id.starts_with(RESPONSE_ID), "responses: {body}");
    let output = body["output"].as_array().expect("an output array");
    assert!(!output.is_empty(), "responses: {body}");
    note("ok: /v1/responses blocking");

    ask["stream"] = json!(true);
    let text = post_stream(addr, "/v1/responses", &ask).await;
    let events = named(&text, "responses stream");
    let names: Vec<&str> = events.iter().map(|(name, _body)| name.as_str()).collect();
    assert_eq!(names.first(), Some(&CREATED), "{text}");
    assert_eq!(names.last(), Some(&COMPLETED), "{text}");
    let ticks: Vec<u64> = events
        .iter()
        .map(|(_name, body)| {
            let tick = body["sequence_number"].as_u64();
            tick.unwrap_or_else(|| panic!("responses stream: no sequence_number: {body}"))
        })
        .collect();
    let rising = ticks.windows(2).all(|pair| pair[0] < pair[1]);
    assert!(
        rising,
        "responses stream: sequence_number not monotonic: {ticks:?}"
    );
    note("ok: /v1/responses streaming, sequence_number monotonic");
}

/// The legacy completions surface, blocking and streamed.
async fn completions(addr: SocketAddr) {
    let mut ask = json!({
        "prompt": "The capital city of France is", "max_tokens": 16, "temperature": 0.0
    });
    let body = post_json(addr, "/v1/completions", &ask).await;
    assert_eq!(body["object"], TEXT_COMPLETION, "completions: {body}");
    let answer = body["choices"][0]["text"].as_str().unwrap_or_default();
    assert!(!answer.is_empty(), "completions: {body}");
    vocabulary(&body["choices"][0]["finish_reason"], &FINISH, "completions");
    counters(&body["usage"], OPENAI_USAGE, "completions");
    note("ok: /v1/completions blocking");

    ask["stream"] = json!(true);
    let text = post_stream(addr, "/v1/completions", &ask).await;
    let chunks = unnamed(&text, "completions stream");
    assert!(!chunks.is_empty(), "completions stream: nothing arrived");
    for chunk in &chunks {
        assert_eq!(
            chunk["object"], TEXT_COMPLETION,
            "completions stream: {chunk}"
        );
    }
    note("ok: /v1/completions streaming");
}

/// The Messages surface, blocking and streamed.
async fn messages(addr: SocketAddr) {
    let mut ask = json!({
        "model": "claude-test",
        "messages": [{"role": "user", "content": "Name the capital city of France."}],
        "max_tokens": 24, "temperature": 0.0
    });
    let body = post_json(addr, "/v1/messages", &ask).await;
    assert_eq!(body["type"], MESSAGE, "messages: {body}");
    assert_eq!(body["role"], ASSISTANT, "messages: {body}");
    let blocks = body["content"].as_array().expect("a content array");
    assert!(!blocks.is_empty(), "messages: {body}");
    for block in blocks {
        assert!(
            block["type"].is_string(),
            "messages: an unlabelled block: {body}"
        );
    }
    vocabulary(&body["stop_reason"], &STOP, "messages");
    counters(&body["usage"], ANTHROPIC_USAGE, "messages");
    note("ok: /v1/messages blocking");

    ask["stream"] = json!(true);
    let text = post_stream(addr, "/v1/messages", &ask).await;
    let events = named(&text, "messages stream");
    let names: Vec<&str> = events.iter().map(|(name, _body)| name.as_str()).collect();
    assert_eq!(names.first(), Some(&MESSAGE_START), "{text}");
    assert_eq!(names.last(), Some(&MESSAGE_STOP), "{text}");
    assert!(
        names.contains(&BLOCK_DELTA),
        "nothing was delivered: {text}"
    );
    note("ok: /v1/messages streaming");
}

/// One refusal in each family's error envelope.
async fn refusals(addr: SocketAddr) {
    let nonsense = json!({"messages": "not a conversation"});
    let path = "/v1/chat/completions";
    let (status, _headers, text) = call(addr, "POST", path, Some(&nonsense)).await;
    assert!(
        status.is_client_error(),
        "chat refusal answered {status}: {text}"
    );
    let body: Value = serde_json::from_str(&text).expect("the refusal is JSON");
    let code = body["error"]["code"].as_str().unwrap_or_default();
    assert!(!code.is_empty(), "chat refusal carries no code: {body}");
    assert_eq!(body["error"]["type"], INVALID, "chat refusal: {body}");
    assert!(
        body.get("type").is_none(),
        "chat refusal in the other shape: {body}"
    );

    let (status, _headers, text) = call(addr, "POST", "/v1/messages", Some(&nonsense)).await;
    assert!(
        status.is_client_error(),
        "messages refusal answered {status}: {text}"
    );
    let body: Value = serde_json::from_str(&text).expect("the refusal is JSON");
    assert_eq!(body["type"], ERROR, "messages refusal: {body}");
    assert_eq!(body["error"]["type"], INVALID, "messages refusal: {body}");
    note(&format!(
        "ok: refusals, OpenAI code {code:?} and the {ERROR} envelope"
    ));
}

/// The routes that answer without generating anything.
async fn plain_routes(addr: SocketAddr) {
    let (status, headers, text) = call(addr, "GET", "/v1/chat/completions", None).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED, "{text}");
    let allow = headers
        .get("allow")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    assert!(
        allow.contains("POST"),
        "the 405 names no method it accepts: {allow:?}"
    );

    let preflight = req(addr, "OPTIONS", "/v1/models")
        .header("origin", "https://example.test")
        .header("access-control-request-method", "GET")
        .body(Body::empty())
        .expect("a preflight builds");
    let (status, headers, _text) = read(send(addr, preflight).await).await;
    assert!(status.is_success(), "the preflight answered {status}");
    assert!(
        headers.contains_key("access-control-allow-methods"),
        "no methods named"
    );
    note(&format!(
        "ok: 405 allows {allow:?}; preflight answered {status}"
    ));

    let (status, _headers, text) = call(addr, "GET", "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let body: Value = serde_json::from_str(&text).expect("the listing is JSON");
    assert_eq!(body["object"], LIST_OBJECT, "{body}");
    assert_eq!(body["data"][0]["object"], MODEL_OBJECT, "{body}");
    let id = body["data"][0]["id"].as_str().unwrap_or_default();
    assert!(!id.is_empty(), "the listing names no model: {body}");

    let (status, _headers, text) = call(addr, "GET", "/cache", None).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let body: Value = serde_json::from_str(&text).expect("the cache report is JSON");
    assert!(
        body["disk"].is_object() && body["engine"].is_object(),
        "{body}"
    );
    note("ok: /v1/models and /cache");
}

/// Two streams, from two surfaces, open on two sockets at once.
async fn interleaved(addr: SocketAddr) {
    let chat = json!({
        "messages": [{"role": "user", "content": "Count slowly from one to twenty."}],
        "max_completion_tokens": 48, "temperature": 0.0, "stream": true
    });
    let message = json!({
        "model": "claude-test",
        "messages": [{"role": "user", "content": "List the days of the week."}],
        "max_tokens": 48, "temperature": 0.0, "stream": true
    });
    let (first, second) = tokio::join!(
        open(addr, "POST", "/v1/chat/completions", Some(&chat)),
        open(addr, "POST", "/v1/messages", Some(&message)),
    );
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(second.status(), StatusCode::OK);
    let mut left: Pin<Box<dyn Stream<Item = _> + Send>> =
        Box::pin(Body::new(first.into_body()).into_data_stream());
    let mut right: Pin<Box<dyn Stream<Item = _> + Send>> =
        Box::pin(Body::new(second.into_body()).into_data_stream());

    let (mut left_text, mut right_text) = (String::new(), String::new());
    let (mut left_done, mut right_done) = (false, false);
    let mut arrivals: Vec<u8> = Vec::new();
    while !(left_done && right_done) {
        tokio::select! {
            chunk = left.next(), if !left_done => match chunk {
                Some(Ok(bytes)) => {
                    arrivals.push(0);
                    left_text.push_str(&String::from_utf8_lossy(&bytes));
                }
                _ended => left_done = true,
            },
            chunk = right.next(), if !right_done => match chunk {
                Some(Ok(bytes)) => {
                    arrivals.push(1);
                    right_text.push_str(&String::from_utf8_lossy(&bytes));
                }
                _ended => right_done = true,
            },
        }
    }

    assert!(
        left_text.contains(DONE),
        "the chat stream did not end: {left_text}"
    );
    let stop = format!("{EVENT}{MESSAGE_STOP}");
    assert!(
        right_text.contains(&stop),
        "the messages stream did not end: {right_text}"
    );
    // Two changes of source is the weakest statement that is still
    // interleaving: one stream got bytes, then the other did, then the first
    // got more. A server that answered them in turn produces exactly one.
    let changes = arrivals
        .windows(2)
        .filter(|pair| pair[0] != pair[1])
        .count();
    let total = arrivals.len();
    assert!(
        changes >= 2,
        "the streams did not interleave: {changes} of {total}"
    );
    note(&format!(
        "ok: interleaved, {changes} source changes over {total} chunks"
    ));
}

/// The whole stack, once.
///
/// One test rather than one per surface because the claim is about a single
/// boot: loading the checkpoint is the expensive part, and per-surface tests
/// would either reload it or share state across cases the harness may reorder.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the local checkpoint and a Metal device"]
async fn the_whole_stack_answers_over_real_http() {
    let Some(model) = model_path() else {
        note("SKIP: no local checkpoint");
        return;
    };
    let Some(config) = config_path() else {
        note("SKIP: no local config beside the checkpoint");
        return;
    };
    if !shiso_metal::metal_available() {
        note("SKIP: no Metal device");
        return;
    }

    let stack = Stack::boot(&model, &config).await;
    note(&format!("booted on http://{}", stack.addr));
    let addr = stack.addr;

    // A ceiling on the session rather than on each request: a device that
    // wedges should end this with a diagnosis, not with a runner still waiting.
    let session = async {
        plain_routes(addr).await;
        chat(addr).await;
        responses(addr).await;
        completions(addr).await;
        messages(addr).await;
        refusals(addr).await;
        interleaved(addr).await;
    };
    let finished = tokio::time::timeout(Duration::from_secs(1200), session).await;

    stack.shutdown().await;
    finished.expect("the session finished inside its deadline");
}
