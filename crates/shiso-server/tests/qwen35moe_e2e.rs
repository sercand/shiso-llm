//! The Qwen 3.5 mixture-of-experts checkpoint, served over a real socket.
//!
//! **Every Qwen3.6-35B-A3B mixture-of-experts number in this repository before this file
//! was measured in process**, through `crates/shiso-bench` or the two `shiso-cuda`
//! cases that drive the model type directly. The model loaded, decoded and took
//! a picture; no server could be pointed at it, because `FamilyDefaults`
//! recognised no `qwen3_5_moe` document and `load_model` had no arm for it.
//! These two cases are the half of that which needs the device: the same boot
//! the binary runs, and two requests over loopback HTTP.
//!
//! `SHISO_CUDA_COMPILE=1 SHISO_TEST_MODEL_QWEN35MOE=<dir> cargo test -p
//! shiso-server --release --features cuda --test qwen35moe_e2e --
//! --ignored --nocapture --test-threads=1`
//!
//! `--test-threads=1` is not optional. Each case here loads 23.4 GB, this box
//! holds one such load at a time, and it runs `earlyoom`: two loads in parallel
//! do not fail, they get killed, and the kill lands on whichever thread the
//! kernel picked rather than on the thing that was wrong.
//!
//! # What is asserted, and what deliberately is not
//!
//! **The served text is a repeated CJK token.** That is this checkpoint's own
//! greedy output, not a defect in this port: it reproduces under `transformers`
//! at the same argmax, and the NVFP4 nibble order, the `lm_head` rows and the
//! FP8 scale direction were each eliminated separately.
//! `crates/shiso-cuda/tests/q35moe_load.rs` grades the logits themselves
//! against that reference to a tenth of a logit, which is where a claim about
//! *correctness* belongs.
//!
//! So these cases assert that the answer is **well-formed** and say nothing
//! about what it says: the status, the envelope, the identifier prefix, the
//! role, a `finish_reason` from the published set, the three counters, and a
//! body that is non-empty. A test that pinned the text would be pinning the
//! checkpoint's opinion, and would have to be widened the day it changed.
//!
//! The one thing the image case does assert beyond well-formedness is that the
//! picture reached the *render* path: the same question with a picture renders
//! to strictly more prompt tokens than without one, because the placeholder run
//! expands to the tower's merged rows. It cannot use this tree's usual
//! paired-pixels assertion - two pictures through one prompt must answer
//! differently - for the reason above: a checkpoint that answers the same
//! repeated token whatever it is shown would fail that assertion while nothing
//! at all was wrong with the splice. The pixels-reach-the-logits claim is
//! `crates/shiso-cuda/tests/q35moe_vision.rs`, which grades the tower's
//! projected rows against `transformers` and runs two pictures through the
//! model type directly.

#![cfg(feature = "cuda")]

use std::io::Cursor;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use hyper_util::rt::TokioIo;
use image::{Rgb, RgbImage};
use serde_json::{Value, json};
use shiso_core::Backend;
use shiso_server::cli::{Cli, Command, ServeArgs};
use shiso_server::router::build_router;
use shiso_server::scheduler::{SchedulerConfig, spawn_engine_with_checkpoints};
use shiso_server::state::AppState;
use shiso_server::vision::VisionTower;

/// The published Chat Completions vocabulary these cases are held to,
/// transcribed from the wire contract rather than read back out of this
/// crate's own types: a test that serializes this server's structs and checks
/// they deserialize passes just as happily when every field name is wrong.
const CHAT_ID: &str = "chatcmpl-";
const CHAT_OBJECT: &str = "chat.completion";
const ASSISTANT: &str = "assistant";
const FINISH: [&str; 5] = [
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
];
const USAGE: [&str; 3] = ["prompt_tokens", "completion_tokens", "total_tokens"];

/// Tokens each request generates.
///
/// Small, and for a reason particular to this checkpoint: what comes back is
/// one token repeated, so a larger budget buys more of the same string and
/// costs decode time on a 35B model with 3B active.
const MAX_TOKENS: u32 = 16;

/// Edge of the picture, in pixels.
///
/// A whole number of merge blocks at this tower's geometry - patch 16, merge 2,
/// so 32 pixels a block - which keeps the resize a no-op and the placeholder
/// run exactly the grid the encoder produces.
const EDGE: u32 = 224;

/// Prints a line of the case's own report.
#[expect(clippy::print_stdout, reason = "a test's report is its output")]
fn note(what: &str) {
    println!("{what}");
}

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// The checkpoint directory, or `None` when it is not on this machine.
///
/// Checked by a file inside it rather than by the directory alone: a snapshot
/// that is still downloading has the directory and not the shards, and a case
/// that skipped on the directory's absence would run and fail on the partial
/// one.
fn snapshot() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("SHISO_TEST_MODEL_QWEN35MOE").ok()?);
    dir.join("model.safetensors.index.json")
        .is_file()
        .then_some(dir)
}

/// Opens device 0, or says on stderr why the case did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open_device() -> Option<Arc<dyn Backend>> {
    match shiso_cuda::CudaBackend::new(0, shiso_cuda::CudaOptions::default()) {
        Ok(cuda) => Some(cuda),
        Err(err) => {
            eprintln!("note: no CUDA device on this machine, skipping: {err}");
            None
        }
    }
}

/// The command line these cases would have been given.
///
/// No `--config`: this family's `--model` is a directory, so `config_path()`
/// resolves `config.json` INSIDE it rather than beside it, and the vocabulary
/// and chat template resolve beside that. Passing one would be testing a path
/// the serve script does not take.
///
/// `--vision` names the same directory `--model` does, because this
/// checkpoint's tower rides in its own shards rather than in a container of its
/// own.
fn args(dir: &Path, vision: bool) -> ServeArgs {
    let model = dir.display().to_string();
    let mut line = vec!["shiso", "serve", "--model", &model, "--ctx", "4096"];
    if vision {
        line.extend_from_slice(&["--vision", &model]);
    }
    let cli = <Cli as clap::Parser>::try_parse_from(line).expect("the command line parses");
    let Command::Serve(serve) = cli.command else {
        panic!("expected Command::Serve");
    };
    *serve
}

/// A running server and the two things that must be torn down, in order.
struct Stack {
    addr: SocketAddr,
    state: Arc<AppState>,
    server: tokio::task::JoinHandle<()>,
    engine: std::thread::JoinHandle<Result<(), shiso_server::scheduler::EngineFailure>>,
}

impl Stack {
    /// Loads the checkpoint - and the tower when one was asked for - starts the
    /// coordinator, and serves on a port the operating system picks.
    ///
    /// The same sequence `shiso_server::serve` runs, through the same factory
    /// and the same family opener. A test that assembled the pieces differently
    /// would not be evidence about the thing that ships.
    async fn boot(backend: &Arc<dyn Backend>, args: ServeArgs) -> Self {
        args.validate().expect("the command line is servable");
        let parsed = shiso_core::HfConfig::from_file(&args.config_path()).expect("the config");
        let shiso_server::model_factory::Loaded {
            model: mut loaded,
            open_tower,
            open_audio: _,
            placement,
            defaults,
            speech: _,
        } = shiso_server::model_factory::load_model(
            backend,
            args.model.as_deref().expect("this test passes --model"),
            &parsed,
            &args,
        )
        .expect("the checkpoint loads");
        assert!(
            !defaults.speculates,
            "this family binds no draft head; `speculates` must stay false or --spec-k \
             stops being refused"
        );

        let vocabulary = Arc::new(
            shiso_server::tokenizer::Vocabulary::from_file(&args.tokenizer_path())
                .expect("the vocabulary loads"),
        );
        let template = std::fs::read_to_string(args.chat_template_path())
            .expect("the checkpoint ships a chat template");
        let mut renderer = shiso_server::tokenizer::render_prompt::PromptRenderer::new(
            &template,
            shiso_server::tokenizer::jinja_helpers::ToolJsonStyle::Compact,
            Arc::clone(&vocabulary),
        )
        .expect("the chat template compiles")
        // Off, and the reason is about what is being measured rather than about
        // this family: with thinking on, a budget of a few dozen tokens is
        // spent entirely inside the thinking block.
        .with_thinking_default(false);

        let tower = args.vision.as_ref().map(|path| {
            let tower = VisionTower::from_tower(
                open_tower(backend, path).expect("the family's tower loads"),
            );
            let hidden = shiso_core::Model::info(loaded.as_mut()).hidden;
            tower
                .check_against(hidden)
                .expect("the tower and the checkpoint are a pair");
            let pads = vocabulary.media_ids(&placement);
            assert!(
                pads.image().is_some(),
                "this vocabulary declares none of this family's image placeholder; \
                 nothing below could splice"
            );
            note(&format!(
                "tower: out_hidden {}, merge {}, marker {}, {} bytes resident",
                tower.out_hidden(),
                tower.spatial_merge_size(),
                tower.placement().image_marker,
                tower.device_bytes()
            ));
            tower
        });
        if let Some(tower) = tower.as_ref() {
            renderer = renderer.with_tower(tower.preprocess(), placement, args.image_limits());
        }

        let model_id = args.model_id();
        let (engine, join) = spawn_engine_with_checkpoints(
            Arc::clone(backend),
            loaded,
            shiso_server::media::MediaTowers {
                vision: tower,
                audio: None,
                cache: None,
            },
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

        // Port zero and then read it back: a fixed port makes this case fail on
        // a machine that happens to be running something else, and 8080 on this
        // box already is.
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

/// Sends one chat request on a connection of its own and returns the body.
async fn post_chat(addr: SocketAddr, body: &Value) -> (StatusCode, String) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("host", addr.to_string())
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("the request builds");
    let socket = tokio::net::TcpStream::connect(addr)
        .await
        .expect("it accepts");
    let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(socket))
        .await
        .expect("the server speaks HTTP/1.1");
    tokio::spawn(async move {
        let _ended = conn.await;
    });
    let response = sender
        .send_request(request)
        .await
        .expect("the server answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(Body::new(response.into_body()), usize::MAX)
        .await
        .expect("the body reads");
    (
        status,
        String::from_utf8(bytes.to_vec()).expect("the body is text"),
    )
}

/// What one answered request yields.
struct Answer {
    text: String,
    prompt_tokens: u64,
}

/// Sends one greedy turn and grades the ENVELOPE, not the words in it.
///
/// Everything checked here is a published fact about the Chat Completions wire
/// format that holds for any model. What the model said is returned for the
/// caller to print, and nothing here compares it to anything.
async fn ask(addr: SocketAddr, content: Value) -> Answer {
    let body = json!({
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": MAX_TOKENS,
        "temperature": 0.0
    });
    let (status, raw) = post_chat(addr, &body).await;
    assert_eq!(status, StatusCode::OK, "the request was refused: {raw}");
    let value: Value = serde_json::from_str(&raw).expect("the answer is JSON");

    assert_eq!(value["object"].as_str(), Some(CHAT_OBJECT), "in {value}");
    let id = value["id"].as_str().unwrap_or_default();
    assert!(id.starts_with(CHAT_ID), "identifier {id:?} in {value}");
    let choice = &value["choices"][0];
    assert_eq!(choice["index"].as_u64(), Some(0), "in {value}");
    assert_eq!(
        choice["message"]["role"].as_str(),
        Some(ASSISTANT),
        "in {value}"
    );
    let finish = choice["finish_reason"].as_str().unwrap_or_default();
    assert!(
        FINISH.contains(&finish),
        "finish_reason {finish:?} is not one of the published set {FINISH:?}"
    );
    for counter in USAGE {
        assert!(
            value["usage"][counter].is_u64(),
            "usage is missing {counter} in {value}"
        );
    }

    let text = choice["message"]["content"]
        .as_str()
        .expect("the turn carries text")
        .to_owned();
    // Non-empty, and nothing more. This checkpoint answers one repeated CJK
    // token; a case that asserted the words would be asserting the
    // checkpoint's opinion rather than this server's behaviour.
    assert!(!text.is_empty(), "an empty answer in {value}");

    let prompt_tokens = value["usage"]["prompt_tokens"]
        .as_u64()
        .expect("a prompt size");
    let completion_tokens = value["usage"]["completion_tokens"]
        .as_u64()
        .expect("a completion size");
    assert!(prompt_tokens > 0, "a prompt of zero tokens in {value}");
    assert!(completion_tokens > 0, "no tokens were generated in {value}");
    assert!(
        completion_tokens <= u64::from(MAX_TOKENS),
        "the server generated {completion_tokens} tokens past a budget of {MAX_TOKENS}"
    );
    assert_eq!(
        value["usage"]["total_tokens"].as_u64(),
        Some(prompt_tokens + completion_tokens),
        "the three counters do not add up in {value}"
    );

    Answer {
        text,
        prompt_tokens,
    }
}

/// One question, as a plain text turn.
fn question() -> Value {
    json!("Describe what you see in one short sentence.")
}

/// The same question with a picture in front of it.
fn question_with(picture: &RgbImage) -> Value {
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(picture.clone())
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("a PNG encodes");
    let uri = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    );
    json!([
        {"type": "image_url", "image_url": {"url": uri}},
        {"type": "text", "text": "Describe what you see in one short sentence."}
    ])
}

/// Four quadrants in four saturated colours.
///
/// Structured rather than flat: a flat field is the one picture whose patches
/// are all identical, and a reader of this file should be able to see what the
/// model was shown.
fn quadrants() -> RgbImage {
    RgbImage::from_fn(EDGE, EDGE, |x, y| {
        let half = EDGE / 2;
        match (x < half, y < half) {
            (true, true) => Rgb([220, 30, 30]),
            (false, true) => Rgb([30, 220, 30]),
            (true, false) => Rgb([30, 30, 220]),
            (false, false) => Rgb([240, 230, 40]),
        }
    })
}

#[test]
#[ignore = "requires a CUDA device and the 23.4 GB checkpoint"]
fn the_served_checkpoint_answers_a_text_completion() {
    let Some(dir) = snapshot() else {
        skip("SHISO_TEST_MODEL_QWEN35MOE names no checkpoint directory");
        return;
    };
    let Some(backend) = open_device() else { return };

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("a runtime")
        .block_on(async {
            let stack = Stack::boot(&backend, args(&dir, false)).await;
            let answer = ask(stack.addr, question()).await;
            note(&format!(
                "text: prompt {} tokens, answer {:?}",
                answer.prompt_tokens, answer.text
            ));
            stack.shutdown().await;
        });
}

#[test]
#[ignore = "requires a CUDA device and the 23.4 GB checkpoint"]
fn the_served_checkpoint_answers_an_image_completion() {
    let Some(dir) = snapshot() else {
        skip("SHISO_TEST_MODEL_QWEN35MOE names no checkpoint directory");
        return;
    };
    let Some(backend) = open_device() else { return };

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("a runtime")
        .block_on(async {
            let stack = Stack::boot(&backend, args(&dir, true)).await;

            // The control first: the same words with no picture, so the
            // comparison below is of one difference and not two.
            let control = ask(stack.addr, question()).await;
            note(&format!(
                "image control (no picture): prompt {} tokens, answer {:?}",
                control.prompt_tokens, control.text
            ));

            let answer = ask(stack.addr, question_with(&quadrants())).await;
            note(&format!(
                "image: prompt {} tokens, answer {:?}",
                answer.prompt_tokens, answer.text
            ));

            // The placeholder run expanded to the tower's merged rows, which is
            // what says the picture reached the render path rather than being
            // dropped with a warning - the exact failure
            // `crates/shiso-server/tests/media_e2e.rs` was written to close for
            // the dense family.
            assert!(
                answer.prompt_tokens > control.prompt_tokens,
                "the picture added no prompt tokens ({} against the text-only {}); the \
                 placeholder run did not expand, so nothing was spliced",
                answer.prompt_tokens,
                control.prompt_tokens
            );
            stack.shutdown().await;
        });
}
