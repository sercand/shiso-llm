//! A picture, over a real socket, all the way to the answer.
//!
//! `media_render.rs` proves the request path: a turn with a picture in it
//! becomes items and a placeholder run of the right length. `e2e.rs` proves the
//! rest of the stack over loopback HTTP. Neither of them ever ran a vision
//! tower, because until now no server constructed one. This is the join: the
//! same boot as `e2e.rs` plus `--vision`, and requests that carry pixels.
//!
//! # The assertion that matters, and the trap it avoids
//!
//! A multimodal test can pass for the wrong reason more easily than almost any
//! other kind. Send one picture, get fluent prose back, assert it is not empty —
//! and the test passes identically on a build that drops the pixels on the
//! floor, because the prompt still carries the placeholder run and the model
//! still answers something. That is not a hypothetical: it is exactly the state
//! this file was written to close, where the scheduler logged a warning and
//! served the text half.
//!
//! So the load-bearing assertion in both cases below is a PAIR of requests:
//!
//! - the same prompt, the same number of media items, the same media
//!   DIMENSIONS — and therefore, token for token, the same rendered prompt and
//!   the same placeholder run;
//! - different PIXELS.
//!
//! Under the text-only path those two requests are byte-identical inputs, so
//! greedy decoding gives them byte-identical answers and the assertion fails.
//! The answers can only differ if the pixels reached the model, and the only
//! route from pixels to the model is the tower, the gather, and the splice this
//! file exists to cover. Nothing about the wording of either answer is asserted,
//! so the test does not depend on the model being good at describing pictures —
//! only on it being affected by them.
//!
//! The recorded answer beside it is a regression lock and nothing more; see the
//! header the fixture carries and `test-vectors/README.md`.
//!
//! # Why the clip is a GIF
//!
//! The plan called the video case ffmpeg-gated. It does not have to be: the
//! media layer sniffs the container and decodes GIF in process, in pure Rust,
//! with no subprocess and no flag. A synthetic clip built here as a GIF
//! therefore exercises the real decode, the real frame sampling and the real
//! temporal grouping on any machine, instead of skipping on every machine that
//! has no decoder installed — which includes this one.

#![cfg(feature = "metal")]

use std::io::Cursor;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use hyper_util::rt::TokioIo;
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use serde_json::{Value, json};
use shiso_core::Backend;
use shiso_server::cli::{Cli, Command, ServeArgs};
use shiso_server::router::build_router;
use shiso_server::scheduler::{SchedulerConfig, spawn_engine_with_checkpoints};
use shiso_server::state::AppState;
use shiso_server::vision::VisionTower;

/// The recorded answers this file is held to.
///
/// Read with `include_str!` rather than from a path at runtime so that a missing
/// fixture is a build failure in the change that removed it, not a skip.
const IMAGE_VECTOR: &str = include_str!("../../../test-vectors/vision/vision-qwen35-image.json");
const VIDEO_VECTOR: &str = include_str!("../../../test-vectors/vision/vision-qwen35-video.json");

/// The recorded answer of the Gemma 4 image case, beside the two Qwen ones.
///
/// Same disclaimer as the others and the same rule: it locks what this machine
/// answered, and the paired-pixels case next to it is what proves the pixels
/// reached the model.
const GEMMA4_IMAGE_VECTOR: &str =
    include_str!("../../../test-vectors/vision/vision-gemma4-image.json");

/// The same lock for the mlx container, which answers with its own weights.
const GEMMA4_MLX_IMAGE_VECTOR: &str =
    include_str!("../../../test-vectors/vision/vision-gemma4-mlx-image.json");

/// Whether this run serves Gemma 4 from the mlx directory rather than the gguf.
fn gemma4_weights_are_mlx() -> bool {
    std::env::var("SHISO_TEST_GEMMA4_WEIGHTS").as_deref() == Ok("mlx")
}

/// Tokens each case generates. Enough to be a sentence, short enough that two
/// device runs of this file are minutes rather than tens of minutes.
const MAX_TOKENS: u32 = 48;

/// Edge of the still images, in pixels.
///
/// A whole number of merge blocks at this tower's geometry (patch 16, merge 2,
/// so 32 pixels a block), which keeps the resize a no-op and the placeholder run
/// exactly the grid the fixture records.
const IMAGE_EDGE: u32 = 224;

/// Edge of the clip's frames, and how many of them.
///
/// Eight frames at two frames a second is four seconds, which the sampler keeps
/// whole, and the tower's temporal patch size folds them into four temporal
/// groups - the multi-group path a still image never reaches.
const CLIP_EDGE: u32 = 128;
const CLIP_FRAMES: u32 = 8;
const CLIP_DELAY_MS: u32 = 500;

/// Prints a line of the test's own report.
#[expect(clippy::print_stdout, reason = "a test's report is its output")]
fn note(what: &str) {
    println!("{what}");
}

// --- what the machine has to provide --------------------------------------

/// The language checkpoint, when the machine has one.
fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL").ok()?);
    path.is_file().then_some(path)
}

/// The checkpoint directory that holds both the architecture config and the
/// vision tower.
fn optiq_dir() -> Option<PathBuf> {
    let var = std::env::var("SHISO_TEST_MODEL_OPTIQ");
    let dir = PathBuf::from(var.ok()?);
    dir.is_dir().then_some(dir)
}

/// The command line this test would have been given.
fn args(model: &Path, optiq: &Path) -> ServeArgs {
    let model = model.display().to_string();
    let config = optiq.join("config.json").display().to_string();
    let vision = optiq.display().to_string();
    let line = [
        "shiso", "serve", "--model", &model, "--config", &config, "--vision", &vision, "--ctx",
        "4096",
    ];
    let cli = <Cli as clap::Parser>::try_parse_from(line).expect("the command line parses");
    let Command::Serve(serve) = cli.command else {
        panic!("expected Command::Serve");
    };
    *serve
}

/// The Gemma 4 language container, when the machine has one.
fn gemma4_model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4").ok()?);
    path.is_file().then_some(path)
}

/// The Gemma 4 sidecar: the MLX export directory that carries the config, the
/// vocabulary and the chat template, none of which the GGUF holds.
fn gemma4_sidecar() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4_SIDECAR").ok()?);
    dir.is_dir().then_some(dir)
}

/// The mmproj this family's tower opens, which sits beside the container.
///
/// Derived from the container's own directory rather than from a second path,
/// because the pair is one download; an explicit
/// `SHISO_TEST_MODEL_GEMMA4_MMPROJ` overrides the guess.
fn gemma4_mmproj(model: &Path) -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("SHISO_TEST_MODEL_GEMMA4_MMPROJ") {
        let path = PathBuf::from(explicit);
        return path.is_file().then_some(path);
    }
    let path = model.with_file_name("gemma-4-E4B-it-mmproj.gguf");
    path.is_file().then_some(path)
}

/// The command line a Gemma 4 boot would have been given: the GGUF, the
/// sidecar's config beside it, and the mmproj as `--vision` - the family's
/// opener takes the tower container's path directly, where the Qwen opener
/// takes the directory that holds the shards.
fn gemma4_args(model: &Path, sidecar: &Path, mmproj: &Path) -> ServeArgs {
    let model = model.display().to_string();
    let config = sidecar.join("config.json").display().to_string();
    let vision = mmproj.display().to_string();
    let line = [
        "shiso", "serve", "--model", &model, "--config", &config, "--vision", &vision, "--ctx",
        "4096",
    ];
    let cli = <Cli as clap::Parser>::try_parse_from(line).expect("the command line parses");
    let Command::Serve(serve) = cli.command else {
        panic!("expected Command::Serve");
    };
    *serve
}

// --- fixtures, generated rather than committed ----------------------------

/// A still image as the data URI a request would carry.
fn png_uri(image: &RgbImage) -> String {
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(image.clone())
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("a PNG encodes");
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    )
}

/// The recorded image: four quadrants in four saturated colours.
///
/// Structured rather than a flat colour, because a flat field is the one picture
/// whose patches are all identical and therefore the one whose encoding is least
/// likely to move an answer. This one differs patch by patch in both axes, which
/// is also what makes the 2D rotation inside the tower observable at all.
fn quadrants() -> RgbImage {
    RgbImage::from_fn(IMAGE_EDGE, IMAGE_EDGE, |x, y| {
        let half = IMAGE_EDGE / 2;
        match (x < half, y < half) {
            (true, true) => Rgb([220, 30, 30]),
            (false, true) => Rgb([30, 220, 30]),
            (true, false) => Rgb([30, 30, 220]),
            (false, false) => Rgb([240, 230, 40]),
        }
    })
}

/// The contrasting image: a fine black-and-white checkerboard.
///
/// The SAME pixel dimensions as [`quadrants`], which is what makes the pair a
/// test of the pixels rather than of the prompt: identical dimensions mean an
/// identical placeholder run, so both requests render to the same tokens.
fn checkerboard() -> RgbImage {
    RgbImage::from_fn(IMAGE_EDGE, IMAGE_EDGE, |x, y| {
        if (x / 16 + y / 16).is_multiple_of(2) {
            Rgb([250, 250, 250])
        } else {
            Rgb([5, 5, 5])
        }
    })
}

/// An animated GIF as the data URI a request would carry.
fn gif_uri(frames: Vec<RgbaImage>) -> String {
    let mut bytes = Vec::new();
    {
        let mut encoder = image::codecs::gif::GifEncoder::new(Cursor::new(&mut bytes));
        encoder
            .set_repeat(image::codecs::gif::Repeat::Infinite)
            .expect("the repeat mode is accepted");
        let delay = image::Delay::from_numer_denom_ms(CLIP_DELAY_MS, 1);
        encoder
            .encode_frames(
                frames
                    .into_iter()
                    .map(|frame| image::Frame::from_parts(frame, 0, 0, delay)),
            )
            .expect("the frames encode");
    }
    format!(
        "data:image/gif;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    )
}

/// The recorded clip: a white bar sweeping left to right across black.
///
/// Motion rather than a still repeated, so that the four temporal groups the
/// sampler produces are four DIFFERENT patch planes. A clip of identical frames
/// would encode to four identical groups and could not distinguish the temporal
/// path from a still image repeated four times.
fn sweeping_bar() -> Vec<RgbaImage> {
    (0..CLIP_FRAMES)
        .map(|frame| {
            let left = frame * (CLIP_EDGE / CLIP_FRAMES);
            RgbaImage::from_fn(CLIP_EDGE, CLIP_EDGE, |x, _y| {
                if x >= left && x < left + CLIP_EDGE / CLIP_FRAMES {
                    Rgba([250, 250, 250, 255])
                } else {
                    Rgba([10, 10, 10, 255])
                }
            })
        })
        .collect()
}

/// The contrasting clip: the whole frame flashing between two colours.
///
/// The same frame count and the same frame size as [`sweeping_bar`], so the two
/// clips occupy the same placeholder run and render to the same prompt.
fn flashing_field() -> Vec<RgbaImage> {
    (0..CLIP_FRAMES)
        .map(|frame| {
            let colour = if frame.is_multiple_of(2) {
                Rgba([230, 40, 40, 255])
            } else {
                Rgba([40, 40, 230, 255])
            };
            RgbaImage::from_pixel(CLIP_EDGE, CLIP_EDGE, colour)
        })
        .collect()
}

// --- the running server ---------------------------------------------------

/// Sends the server's log to the test's output.
///
/// A 500 says only "internal server error" on the wire; the cause is in the
/// log, and a failing device case with no log is a case nobody can read.
/// Another test in the binary may have installed one already.
fn install_log() {
    let _installed = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_test_writer()
        .try_init();
}

/// The renderer with the audio tower attached, as the binary's boot does.
/// The towers the engine is spawned with; no encode cache, so every item is
/// encoded afresh and the two answers compared are the towers' own.
fn towers(
    vision: VisionTower,
    audio: Option<Arc<dyn shiso_audio::AudioTower>>,
) -> shiso_server::media::MediaTowers {
    shiso_server::media::MediaTowers {
        vision: Some(vision),
        audio,
        cache: None,
    }
}

fn with_audio(
    renderer: shiso_server::tokenizer::render_prompt::PromptRenderer,
    audio: Option<Arc<dyn shiso_audio::AudioTower>>,
    placement: shiso_vision::MediaPlacement,
    args: &ServeArgs,
) -> shiso_server::tokenizer::render_prompt::PromptRenderer {
    match audio {
        Some(tower) => renderer.with_audio_tower(
            tower,
            placement,
            args.audio_limits(),
            args.audio_decode_policy(),
        ),
        None => renderer,
    }
}

/// A booted stack, and the two things that must be torn down, in order.
struct Stack {
    addr: SocketAddr,
    state: Arc<AppState>,
    server: tokio::task::JoinHandle<()>,
    engine: std::thread::JoinHandle<Result<(), shiso_server::scheduler::EngineFailure>>,
}

impl Stack {
    /// Loads the checkpoint AND the tower, starts the coordinator, and serves
    /// on a port the operating system picks.
    ///
    /// The same sequence the binary's own boot path runs, including the tower's
    /// width check against the language model, because a test that assembled the
    /// pieces differently would not be evidence about the thing that ships.
    async fn boot(model: &Path, optiq: &Path) -> Self {
        Self::boot_args(args(model, optiq)).await
    }

    /// The same boot from an already-parsed command line, so a second family
    /// comes through the identical door rather than beside it.
    async fn boot_args(args: ServeArgs) -> Self {
        install_log();
        args.validate().expect("the command line is servable");
        let backend: Arc<dyn Backend> = shiso_metal::MetalBackend::new().expect("a Metal device");
        let parsed = shiso_core::HfConfig::from_file(&args.config_path()).expect("the config");
        let shiso_server::model_factory::Loaded {
            model: mut loaded,
            open_tower,
            open_audio,
            placement,
            defaults,
        } = shiso_server::model_factory::load_model(
            &backend,
            args.model.as_deref().expect("this test passes --model"),
            &parsed,
            &args,
        )
        .expect("the checkpoint loads");

        let vision = args.vision.as_ref().expect("this test passes --vision");
        // Through the family's own opener, which is the door the binary uses:
        // a test that constructed the tower some other way would not be
        // evidence about the thing that ships.
        let tower = VisionTower::from_tower(
            open_tower(&backend, vision).expect("the family's tower loads"),
        );
        let hidden = shiso_core::Model::info(loaded.as_mut()).hidden;
        tower
            .check_against(hidden)
            .expect("the tower and the checkpoint are a pair");
        note(&format!(
            "tower: out_hidden {}, merge {}, marker {}, {} bytes resident",
            tower.out_hidden(),
            tower.spatial_merge_size(),
            tower.placement().image_marker,
            tower.device_bytes()
        ));

        // The binary's own opener, width and media-path checks included: a copy
        // here would pass while the shipped one drifted.
        let audio = shiso_server::open_audio_tower(&backend, &args, loaded.as_mut(), open_audio)
            .expect("the audio tower opens or is absent");
        note(&format!("audio tower resident: {}", audio.is_some()));

        let vocabulary = Arc::new(
            shiso_server::tokenizer::Vocabulary::from_file(&args.tokenizer_path())
                .expect("the vocabulary loads"),
        );
        let pads = vocabulary.media_ids(&placement);
        assert!(
            pads.image().is_some(),
            "this vocabulary declares none of this family's image placeholder; \
             nothing below could splice"
        );
        if placement.video_marker.is_some() {
            assert!(
                pads.video().is_some(),
                "this family renders clips and this vocabulary does not declare the clip \
                 placeholder; nothing below could splice"
            );
        }
        let template = std::fs::read_to_string(args.chat_template_path())
            .expect("the checkpoint ships a chat template");
        let renderer = shiso_server::tokenizer::render_prompt::PromptRenderer::new(
            &template,
            shiso_server::tokenizer::jinja_helpers::ToolJsonStyle::Compact,
            Arc::clone(&vocabulary),
        )
        .expect("the chat template compiles")
        // From the tower rather than from the config document: the tower is the
        // thing the patch rows are about to be fed to, and this is the path the
        // binary takes when one is open.
        .with_tower(tower.preprocess(), placement, args.image_limits())
        .with_video(args.video_decode_policy(), args.video_fps)
        // Thinking off, and the reason is about what is being measured: with it
        // on, a budget of a few dozen tokens is spent entirely inside the
        // thinking block, and what the pair of answers would be compared on is
        // the model's preamble rather than its description. The multimodal path
        // is identical either way.
        .with_thinking_default(false);
        let renderer = with_audio(renderer, audio.clone(), placement, &args);

        let model_id = args.model_id();
        let (engine, join) = spawn_engine_with_checkpoints(
            Arc::clone(&backend),
            loaded,
            towers(tower, audio),
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

/// Sends one chat request on a connection of its own and returns the JSON body.
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

/// What one answered request yields: the assistant's text and the prompt size.
struct Answer {
    text: String,
    prompt_tokens: u64,
    /// The prompt's audio soft tokens, as the usage block reports them.
    audio_tokens: u64,
}

/// One greedy turn over a media part and a question.
async fn ask(addr: SocketAddr, part: Value, question: &str) -> Answer {
    let body = json!({
        "messages": [{"role": "user", "content": [part, {"type": "text", "text": question}]}],
        "max_completion_tokens": MAX_TOKENS,
        "temperature": 0.0
    });
    let (status, text) = post_chat(addr, &body).await;
    assert_eq!(status, StatusCode::OK, "the request was refused: {text}");
    let value: Value = serde_json::from_str(&text).expect("the answer is JSON");
    let answer = value["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert!(!answer.is_empty(), "an empty answer in {value}");
    Answer {
        text: answer,
        prompt_tokens: value["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        audio_tokens: value["usage"]["prompt_tokens_details"]["audio_tokens"]
            .as_u64()
            .unwrap_or(0),
    }
}

/// The recorded answer and prompt size a fixture carries.
fn recorded(vector: &str) -> (String, u64) {
    let value: Value = serde_json::from_str(vector).expect("the fixture is JSON");
    let text = value["answer"]
        .as_str()
        .expect("the fixture records an answer")
        .to_owned();
    let tokens = value["prompt_tokens"]
        .as_u64()
        .expect("the fixture records a prompt size");
    (text, tokens)
}

/// Compares an answer with its recorded lock, printing both on a mismatch.
///
/// The recorded value is never widened to make a run land: it was captured on
/// this machine by this implementation, so a disagreement means this build
/// answers differently from the build that recorded it, and that is the finding.
fn locks(case: &str, got: &Answer, vector: &str) {
    let (want_text, want_tokens) = recorded(vector);
    assert_eq!(
        got.prompt_tokens, want_tokens,
        "{case}: the prompt is {} tokens but the fixture records {want_tokens}; the \
         placeholder run moved, which is a render-path change and not a decode one",
        got.prompt_tokens
    );
    assert_eq!(
        got.text.as_str(),
        want_text.as_str(),
        "{case}: this build answers differently from the build that recorded the lock"
    );
}

// --- the cases ------------------------------------------------------------

/// One greedy turn with no media at all.
///
/// The control. A server that has a tower must still take the text path for a
/// request that carries no pictures - same launches, same one-dimensional
/// rotation, no splice - and this is the cheapest statement that the tower's
/// presence did not change what a text request costs or answers.
async fn text_only(addr: SocketAddr) -> Answer {
    let body = json!({
        "messages": [{"role": "user", "content": "Name the capital city of France."}],
        "max_completion_tokens": MAX_TOKENS,
        "temperature": 0.0
    });
    let (status, text) = post_chat(addr, &body).await;
    assert_eq!(status, StatusCode::OK, "the request was refused: {text}");
    let value: Value = serde_json::from_str(&text).expect("the answer is JSON");
    let answer = value["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert!(!answer.is_empty(), "an empty answer in {value}");
    Answer {
        text: answer,
        prompt_tokens: value["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        audio_tokens: value["usage"]["prompt_tokens_details"]["audio_tokens"]
            .as_u64()
            .unwrap_or(0),
    }
}

/// Two pictures, one prompt, and the answers that have to differ.
async fn image_case(addr: SocketAddr) {
    let control = text_only(addr).await;
    note(&format!(
        "text-only control: prompt {} tokens, answer {:?}",
        control.prompt_tokens, control.text
    ));

    let question = "Describe this image in one short sentence.";
    let part = |uri: String| json!({"type": "image_url", "image_url": {"url": uri}});

    let recorded_case = ask(addr, part(png_uri(&quadrants())), question).await;
    note(&format!(
        "image/quadrants: prompt {} tokens, answer {:?}",
        recorded_case.prompt_tokens, recorded_case.text
    ));
    let contrast = ask(addr, part(png_uri(&checkerboard())), question).await;
    note(&format!(
        "image/checkerboard: prompt {} tokens, answer {:?}",
        contrast.prompt_tokens, contrast.text
    ));

    // The pair. Same question, same pixel dimensions, therefore the same
    // rendered prompt and the same placeholder run - which is what the equal
    // prompt sizes below assert, so that a difference in the answers cannot be
    // explained by a difference in the prompt.
    assert_eq!(
        recorded_case.prompt_tokens, contrast.prompt_tokens,
        "the two requests did not render to the same prompt length, so a difference \
         in their answers would prove nothing"
    );
    assert_ne!(
        recorded_case.text, contrast.text,
        "two different pictures produced the same answer from the same prompt: the \
         pixels did not reach the model"
    );
    note("ok: two pictures, one prompt, two answers - the pixels reached the model");

    // Determinism, which is what makes the lock below a fact rather than a
    // sample: greedy decoding over the same bytes twice.
    let again = ask(addr, part(png_uri(&quadrants())), question).await;
    assert_eq!(
        again.text, recorded_case.text,
        "two identical multimodal requests answered differently"
    );
    note("ok: the same picture twice, the same answer");

    locks("image", &recorded_case, IMAGE_VECTOR);
    note("ok: the recorded image answer replays");
}

/// Two clips, one prompt, and the answers that have to differ.
async fn video_case(addr: SocketAddr) {
    let question = "Describe what happens in this clip in one short sentence.";
    let part = |uri: String| json!({"type": "video_url", "video_url": {"url": uri}});

    let recorded_case = ask(addr, part(gif_uri(sweeping_bar())), question).await;
    note(&format!(
        "video/sweeping-bar: prompt {} tokens, answer {:?}",
        recorded_case.prompt_tokens, recorded_case.text
    ));
    let contrast = ask(addr, part(gif_uri(flashing_field())), question).await;
    note(&format!(
        "video/flashing-field: prompt {} tokens, answer {:?}",
        contrast.prompt_tokens, contrast.text
    ));

    assert_eq!(
        recorded_case.prompt_tokens, contrast.prompt_tokens,
        "the two clips did not render to the same prompt length"
    );
    assert_ne!(
        recorded_case.text, contrast.text,
        "two different clips produced the same answer from the same prompt: the frames \
         did not reach the model"
    );
    note("ok: two clips, one prompt, two answers - the frames reached the model");

    locks("video", &recorded_case, VIDEO_VECTOR);
    note("ok: the recorded clip answer replays");
}

/// The paired-pixels pair, run for the Gemma 4 boot: same question, same
/// dimensions, different pixels, and the answers that therefore must differ.
///
/// The reasoning is the one at the head of this file, and it is the family's
/// tower this time. It returns the recorded half's answer so the golden case
/// beside it can lock it without a third request.
async fn gemma4_paired_case(addr: SocketAddr) -> Answer {
    let question = "Describe this image in one short sentence.";
    let part = |uri: String| json!({"type": "image_url", "image_url": {"url": uri}});

    let recorded_case = ask(addr, part(png_uri(&quadrants())), question).await;
    note(&format!(
        "gemma4/image/quadrants: prompt {} tokens, answer {:?}",
        recorded_case.prompt_tokens, recorded_case.text
    ));
    let contrast = ask(addr, part(png_uri(&checkerboard())), question).await;
    note(&format!(
        "gemma4/image/checkerboard: prompt {} tokens, answer {:?}",
        contrast.prompt_tokens, contrast.text
    ));

    assert_eq!(
        recorded_case.prompt_tokens, contrast.prompt_tokens,
        "the two requests did not render to the same prompt length, so a difference \
         in their answers would prove nothing"
    );
    assert_ne!(
        recorded_case.text, contrast.text,
        "two different pictures produced the same answer from the same prompt: the \
         pixels did not reach the model"
    );
    note("ok: two pictures, one prompt, two answers - the pixels reached the model");
    recorded_case
}

/// The Gemma 4 golden: the pair above, determinism, and the recorded answer.
async fn gemma4_image_case(addr: SocketAddr) {
    let control = text_only(addr).await;
    note(&format!(
        "gemma4 text-only control: prompt {} tokens, answer {:?}",
        control.prompt_tokens, control.text
    ));

    let recorded_case = gemma4_paired_case(addr).await;

    let question = "Describe this image in one short sentence.";
    let part = |uri: String| json!({"type": "image_url", "image_url": {"url": uri}});
    let again = ask(addr, part(png_uri(&quadrants())), question).await;
    assert_eq!(
        again.text, recorded_case.text,
        "two identical multimodal requests answered differently"
    );
    // Two containers, two locks. The mlx export is a different quantization of
    // the same model and answers differently, so grading it against the gguf
    // answer would fail on the checkpoint rather than on this build - and
    // grading it against nothing, which is what this did, left the only path
    // through the mlx tower with no recorded expectation at all.
    if gemma4_weights_are_mlx() {
        locks("gemma4/image/mlx", &recorded_case, GEMMA4_MLX_IMAGE_VECTOR);
        note("ok: the recorded mlx image answer replays");
    } else {
        locks("gemma4/image", &recorded_case, GEMMA4_IMAGE_VECTOR);
        note("ok: the recorded image answer replays");
    }
}

// --- the entry points -----------------------------------------------------

/// A request carrying an image produces an answer that depended on the image.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the local checkpoint, the local tower and a Metal device"]
async fn image_e2e_golden() {
    let Some(stack) = booted().await else {
        return;
    };
    let addr = stack.addr;
    let finished =
        tokio::time::timeout(Duration::from_secs(1200), async { image_case(addr).await }).await;
    stack.shutdown().await;
    finished.expect("the image case finished inside its deadline");
}

/// A request carrying a synthetic clip produces an answer that depended on it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the local checkpoint, the local tower and a Metal device"]
async fn video_e2e_synthetic() {
    let Some(stack) = booted().await else {
        return;
    };
    let addr = stack.addr;
    let finished =
        tokio::time::timeout(Duration::from_secs(1200), async { video_case(addr).await }).await;
    stack.shutdown().await;
    finished.expect("the video case finished inside its deadline");
}

/// The booted stack, or `None` having printed why this machine cannot run it.
///
/// `None` is a skip rather than a failure: the checkpoint, the tower and the
/// device are all local facts, and a machine without them has nothing to say
/// about this code either way.
async fn booted() -> Option<Stack> {
    let Some(model) = model_path() else {
        note("SKIP: no local checkpoint");
        return None;
    };
    let Some(optiq) = optiq_dir() else {
        note("SKIP: no local checkpoint directory for the config and the tower");
        return None;
    };
    if !shiso_metal::metal_available() {
        note("SKIP: no Metal device");
        return None;
    }
    let stack = Stack::boot(&model, &optiq).await;
    note(&format!("booted on http://{}", stack.addr));
    Some(stack)
}

/// Two pictures of one size and one prompt answer differently, through the
/// Gemma 4 tower: the pixels reached the model.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the local Gemma 4 container, mmproj and a Metal device"]
async fn gemma4_paired_pixels_differ() {
    let Some(stack) = booted_gemma4().await else {
        return;
    };
    let addr = stack.addr;
    let finished = tokio::time::timeout(Duration::from_secs(1200), async {
        gemma4_paired_case(addr).await
    })
    .await;
    stack.shutdown().await;
    finished.expect("the paired image case finished inside its deadline");
}

/// And the recorded Gemma 4 answer replays, prompt length and all.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the local Gemma 4 container, mmproj and a Metal device"]
async fn gemma4_image_e2e_golden() {
    let Some(stack) = booted_gemma4().await else {
        return;
    };
    let addr = stack.addr;
    let finished = tokio::time::timeout(Duration::from_secs(1200), async {
        gemma4_image_case(addr).await;
    })
    .await;
    stack.shutdown().await;
    finished.expect("the golden image case finished inside its deadline");
}

/// The spoken clip, read from the committed audio fixtures.
fn speech_en_wav() -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/audio/clips/speech-en.wav");
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A spoken clip reaches the model: asked where the speaker went, it names the
/// city the speech names. The clip is thirty seconds of a public address that
/// says "Chicago" once; no word of the prompt does.
async fn gemma4_audio_case(addr: SocketAddr) {
    use base64::Engine as _;
    let data = base64::engine::general_purpose::STANDARD.encode(speech_en_wav());
    let part = json!({"type": "input_audio", "input_audio": {"data": data, "format": "wav"}});
    let answer = ask(
        addr,
        part,
        "What city did the speaker travel to? Answer with one word.",
    )
    .await;
    note(&format!(
        "gemma4/audio/speech-en: prompt {} tokens, answer {:?}",
        answer.prompt_tokens, answer.text
    ));
    // 750 soft tokens for thirty seconds, plus the bracket and the words.
    assert!(
        answer.prompt_tokens > 750,
        "the clip's pad run is missing from the prompt: {} tokens",
        answer.prompt_tokens
    );
    assert_eq!(
        answer.audio_tokens, 750,
        "thirty seconds is 750 audio soft tokens in the usage block"
    );
    assert!(
        answer.text.contains("Chicago"),
        "the answer does not name the city the speech names: {:?}",
        answer.text
    );
    note("ok: the speech reached the model");

    // A picture before the clip in the same turn: two items of two kinds,
    // encoded by two towers and gathered into one block. Rows spliced in the
    // wrong order would put the picture's rows in the clip's run.
    let picture = json!({"type": "image_url", "image_url": {"url": png_uri(&quadrants())}});
    let clip = json!({"type": "input_audio", "input_audio": {"data": data, "format": "wav"}});
    let body = json!({
        "messages": [{"role": "user", "content": [
            picture,
            clip,
            {"type": "text", "text": "Ignore the picture. What city did the speaker travel to? Answer with one word."},
        ]}],
        "max_completion_tokens": MAX_TOKENS,
        "temperature": 0.0
    });
    let (status, text) = post_chat(addr, &body).await;
    assert_eq!(status, StatusCode::OK, "the request was refused: {text}");
    let value: Value = serde_json::from_str(&text).expect("the answer is JSON");
    let mixed = value["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    note(&format!(
        "gemma4/audio/image+speech-en: prompt {} tokens, answer {mixed:?}",
        value["usage"]["prompt_tokens"]
    ));
    assert_eq!(value["usage"]["prompt_tokens_details"]["audio_tokens"], 750);
    assert!(
        value["usage"]["prompt_tokens"].as_u64().unwrap_or(0) > answer.prompt_tokens,
        "the picture's run is missing from the prompt"
    );
    assert!(
        mixed.contains("Chicago"),
        "with a picture before it, the clip's rows did not land in the clip's run: {mixed:?}"
    );
    note("ok: a picture and a clip in one turn, each in its own run");
}

/// A request carrying speech is answered from what was said.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the local Gemma 4 container, mmproj and a Metal device"]
async fn gemma4_audio_e2e_names_the_city() {
    let Some(stack) = booted_gemma4().await else {
        return;
    };
    let addr = stack.addr;
    let finished = tokio::time::timeout(Duration::from_secs(1200), async {
        gemma4_audio_case(addr).await;
    })
    .await;
    stack.shutdown().await;
    finished.expect("the audio case finished inside its deadline");
}

/// The booted Gemma 4 stack, or `None` having printed why this machine cannot
/// run it. The same skip discipline as [`booted`]: container, mmproj and device
/// are all local facts.
async fn booted_gemma4() -> Option<Stack> {
    let Some(sidecar) = gemma4_sidecar() else {
        note("SKIP: no local Gemma 4 sidecar for the config, vocabulary and template");
        return None;
    };
    let (model, mmproj) = if gemma4_weights_are_mlx() {
        (sidecar.clone(), sidecar.clone())
    } else {
        let Some(model) = gemma4_model_path() else {
            note("SKIP: no local Gemma 4 container");
            return None;
        };
        let Some(mmproj) = gemma4_mmproj(&model) else {
            note("SKIP: no Gemma 4 mmproj beside the container");
            return None;
        };
        (model, mmproj)
    };
    if !shiso_metal::metal_available() {
        note("SKIP: no Metal device");
        return None;
    }
    let args = gemma4_args(&model, &sidecar, &mmproj);
    let stack = Stack::boot_args(args).await;
    note(&format!("gemma4 booted on http://{}", stack.addr));
    Some(stack)
}
