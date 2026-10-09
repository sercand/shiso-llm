//! The image generation surface, driven through the router with a mocked
//! generator: the statuses a client fixes on its own, the buffered body, the
//! SSE frame order, and the size grammar.
//!
//! The generator is mocked (the `image_service` tests' trick): what is under
//! test is the wire contract - refusals, normalization, base64 round-trip,
//! frame order - not the diffusion math, which needs a device and belongs to
//! `shiso-models`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use serde_json::json;
use shiso_core::ids::Seed;
use shiso_server::image::size::normalize_size;
use shiso_server::image::{ImageService, ImageServiceDesc};
use shiso_server::router::build_router;
use shiso_server::state::AppState;
use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageLimits,
    ImageOutputFormat, ImageStepEvent,
};
use tower::ServiceExt as _;

/// The encoded bytes the mock hands back: a PNG signature, no pixels.
/// The grammar read against the flux-like limits.
fn normalize_size_default(value: &str) -> Result<(u32, u32), shiso_server::image::size::SizeError> {
    normalize_size(value, &FLUX_LIKE)
}

fn png_bytes() -> Vec<u8> {
    vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
}

/// A servable job that is not the request under test: how the worker and
/// the queue are occupied before a request arrives.
fn filler_job() -> ImageGenerationJob {
    ImageGenerationJob {
        prompt: "filler".to_owned(),
        width: 256,
        height: 256,
        steps: 1,
        seed: Seed::new(1),
        output_format: ImageOutputFormat::Png,
        output_compression: None,
        input_references: Vec::new(),
    }
}

/// A generator that records every job it is handed, so a test can read the
/// dimensions and seeds the wire layer actually produced, and can emit partial
/// step images when the request asked to stream.
#[derive(Debug)]
struct RecordingGenerator {
    jobs: Mutex<Vec<ImageGenerationJob>>,
    partials: bool,
    limits: ImageLimits,
}

/// The limits the flux-like mocks answer with: the 16 grid, 256..=2048, 1024
/// square at 4 steps.
const FLUX_LIKE: ImageLimits = ImageLimits {
    size_multiple: 16,
    min_side: 256,
    max_side: 2048,
    default_size: (1024, 1024),
    default_steps: 4,
    fixed_steps: None,
};

impl RecordingGenerator {
    fn new(partials: bool, limits: ImageLimits) -> Self {
        Self {
            jobs: Mutex::new(Vec::new()),
            partials,
            limits,
        }
    }

    fn recorded(&self) -> Vec<ImageGenerationJob> {
        self.jobs
            .lock()
            .expect("the log lock is never poisoned")
            .clone()
    }
}

impl ImageGenerator for RecordingGenerator {
    fn model_id(&self) -> &'static str {
        "mock-flux"
    }

    fn limits(&self) -> ImageLimits {
        self.limits
    }

    fn generate(
        &self,
        job: &ImageGenerationJob,
        progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        self.jobs
            .lock()
            .expect("the log lock is never poisoned")
            .push(job.clone());
        if self.partials
            && let Some(progress) = progress
        {
            for step in 0..job.steps {
                let mark = u8::try_from(step).unwrap_or(b'0');
                progress(ImageStepEvent {
                    step,
                    total_steps: job.steps,
                    partial_image: Some(vec![mark; 4]),
                });
            }
        }
        Ok(GeneratedImage {
            bytes: png_bytes(),
            media_type: "image/png",
            width: job.width,
            height: job.height,
            prompt_tokens: 10,
            completion_tokens: 1024,
        })
    }
}

/// A generator that fails the second job it is handed: how a failure after
/// the stream has spent its 200 is made to happen. The mock seam is the
/// generator's return value, so the failure lands between two images; a
/// failure inside one generation rides the same handler arm.
#[derive(Debug)]
struct FailingSecondGenerator {
    calls: AtomicUsize,
}

impl ImageGenerator for FailingSecondGenerator {
    fn model_id(&self) -> &'static str {
        "mock-flux"
    }

    fn limits(&self) -> ImageLimits {
        FLUX_LIKE
    }

    fn generate(
        &self,
        job: &ImageGenerationJob,
        _progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 1 {
            return Err(ImageGenError::Backend(
                "the second job always fails".to_owned(),
            ));
        }
        Ok(GeneratedImage {
            bytes: png_bytes(),
            media_type: "image/png",
            width: job.width,
            height: job.height,
            prompt_tokens: 10,
            completion_tokens: 1024,
        })
    }
}

/// A generator that reports it has started and then waits until the test
/// releases it, emitting no step events at all: how a stream's head is
/// caught while the first generation is still running and no partial
/// exists yet.
#[derive(Debug)]
struct GateGenerator {
    started: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    gate: Mutex<mpsc::Receiver<()>>,
}

impl ImageGenerator for GateGenerator {
    fn model_id(&self) -> &'static str {
        "mock-flux"
    }

    fn limits(&self) -> ImageLimits {
        FLUX_LIKE
    }

    fn generate(
        &self,
        job: &ImageGenerationJob,
        _progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        if let Some(sender) = self
            .started
            .lock()
            .expect("the started lock is never poisoned")
            .take()
        {
            let _gone = sender.send(());
        }
        // Waits for the test's token; when the test drops the sender the
        // wait ends and the job completes.
        let _gone = self
            .gate
            .lock()
            .expect("the gate lock is never poisoned")
            .recv();
        Ok(GeneratedImage {
            bytes: png_bytes(),
            media_type: "image/png",
            width: job.width,
            height: job.height,
            prompt_tokens: 10,
            completion_tokens: 1024,
        })
    }
}

/// The state for one test: a server with no language model and the mock
/// generator attached, plus the log the generator writes into.
fn state_with_mock(partials: bool) -> (Arc<AppState>, Arc<RecordingGenerator>) {
    state_with_limits(partials, FLUX_LIKE)
}

/// The same, for a model whose limits are `limits`.
fn state_with_limits(
    partials: bool,
    limits: ImageLimits,
) -> (Arc<AppState>, Arc<RecordingGenerator>) {
    let generator = Arc::new(RecordingGenerator::new(partials, limits));
    let concrete = Arc::clone(&generator);
    let boxed: Arc<dyn ImageGenerator> = concrete;
    let service = ImageService::start(ImageServiceDesc {
        generator: boxed,
        model_id: "mock-flux".to_owned(),
        max_queue: 8,
    })
    .expect("starts image service");
    let state = Arc::new(AppState::without_language_model().with_image(service));
    (state, generator)
}

/// POSTs one JSON body at one image route and reads the whole answer.
async fn post_json(
    state: Arc<AppState>,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let response = build_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("builds"),
        )
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

/// GETs `/v1/models` and reads the listed ids.
async fn model_ids(state: Arc<AppState>) -> Vec<String> {
    let response = build_router(state)
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("builds"),
        )
        .await
        .expect("the router answers");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
    body["data"]
        .as_array()
        .expect("a list")
        .iter()
        .filter_map(|entry| entry["id"].as_str())
        .map(str::to_owned)
        .collect()
}

// ---------------------------------------------------------------------------
// The brief's two acceptance tests.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn images_endpoint_returns_503_when_no_image_model() {
    // All three spellings of the route are one handler and answer alike:
    // a well-formed request this server could have run, refused because
    // no generator is loaded.
    for uri in ["/images", "/v1/images", "/v1/images/generations"] {
        let state = Arc::new(AppState::without_language_model());
        let (status, body) = post_json(state, uri, json!({"prompt": "a futuristic city"})).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{uri}: {body}");
        assert_eq!(
            body["error"]["code"], "image_model_not_loaded",
            "{uri}: {body}"
        );
    }
}

#[tokio::test]
async fn images_endpoint_rejects_unsupported_url_response_format() {
    // Refused before the image-model question: a body this server will never
    // act on is the caller's fault whether or not a generator is loaded.
    let state = Arc::new(AppState::without_language_model());
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a futuristic city", "response_format": "url"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "unsupported_parameter", "{body}");
    assert_eq!(body["error"]["param"], "response_format", "{body}");
}

#[tokio::test]
async fn a_malformed_size_is_refused_before_the_image_model_question() {
    // The same rule as the response_format refusal, on the field the plan
    // checks rather than the wire struct: the wire is validated into a plan
    // before the service lookup, so a malformed size is a 400 even with no
    // generator loaded - the 503 is reserved for a well-formed request.
    let state = Arc::new(AppState::without_language_model());
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door", "size": "banana"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "size", "{body}");
}

// ---------------------------------------------------------------------------
// The refusals a caller fixes on its own.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_unencodable_output_format_is_refused() {
    let (state, _log) = state_with_mock(false);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door", "output_format": "webp"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "output_format", "{body}");
}

#[tokio::test]
async fn an_n_outside_one_to_ten_is_refused() {
    for n in [0, 11] {
        let (state, _log) = state_with_mock(false);
        let (status, body) = post_json(
            state,
            "/v1/images/generations",
            json!({"prompt": "a red door", "n": n}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "n={n}: {body}");
        assert_eq!(body["error"]["param"], "n", "n={n}: {body}");
    }
}

#[tokio::test]
async fn a_malformed_size_string_is_refused() {
    for size in ["banana", "1024", "1024xbanana", "13:7"] {
        let (state, _log) = state_with_mock(false);
        let (status, body) = post_json(
            state,
            "/v1/images/generations",
            json!({"prompt": "a red door", "size": size}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "size={size:?}: {body}");
        assert_eq!(body["error"]["param"], "size", "size={size:?}: {body}");
    }
}

#[tokio::test]
async fn a_named_resolution_is_normalized_to_its_square() {
    // "2K" names a resolution class, and the class has one honest answer on
    // this surface: the square of 2*1024, read through the same rounding and
    // clamping as any other size. "4K" names 4096 and lands on the ceiling.
    let (state, log) = state_with_mock(false);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door", "size": "2K"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let jobs = log.recorded();
    assert_eq!((jobs[0].width, jobs[0].height), (2048, 2048));

    let (state, log) = state_with_mock(false);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door", "size": "4k"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let jobs = log.recorded();
    assert_eq!(
        (jobs[0].width, jobs[0].height),
        (2048, 2048),
        "4K clamps to the servable ceiling rather than being refused"
    );
}

#[tokio::test]
async fn an_auto_size_is_an_omitted_size() {
    // "auto" is the caller asking for the server's default, so it gets what
    // an omitted size gets; and where a default is not the question - an
    // aspect ratio is named - it steps aside exactly as an omitted size does.
    let (state, log) = state_with_mock(false);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door", "size": "auto"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let jobs = log.recorded();
    assert_eq!((jobs[0].width, jobs[0].height), (1024, 1024));

    let (state, log) = state_with_mock(false);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door", "size": "AUTO", "aspect_ratio": "16:9"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let jobs = log.recorded();
    assert_eq!((jobs[0].width, jobs[0].height), (1344, 768));
}

#[tokio::test]
async fn a_compression_outside_one_to_hundred_is_refused() {
    // Zero is refused, not clamped: a quality of zero has no encoding to
    // mean it, and silently encoding at quality 1 would answer a request
    // nobody made.
    for compression in [0, 101] {
        let (state, _log) = state_with_mock(false);
        let (status, body) = post_json(
            state,
            "/v1/images/generations",
            json!({"prompt": "a red door", "output_compression": compression}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "c={compression}: {body}");
        assert_eq!(
            body["error"]["param"], "output_compression",
            "c={compression}: {body}"
        );
    }
}

// ---------------------------------------------------------------------------
// What the wire values become: the job the generator is handed.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_aspect_ratio_becomes_the_table_dimensions() {
    let (state, log) = state_with_mock(false);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a wide vista", "aspect_ratio": "16:9"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let jobs = log.recorded();
    assert_eq!(jobs.len(), 1);
    assert_eq!((jobs[0].width, jobs[0].height), (1344, 768));
}

#[tokio::test]
async fn a_nonstandard_size_is_rounded_to_the_grid() {
    let (state, log) = state_with_mock(false);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door", "size": "1000x1000"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let jobs = log.recorded();
    assert_eq!((jobs[0].width, jobs[0].height), (1008, 1008));
}

#[tokio::test]
async fn a_request_that_names_nothing_gets_the_server_defaults() {
    let (state, log) = state_with_mock(false);
    let (status, body) = post_json(state, "/images", json!({"prompt": "a red door"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let jobs = log.recorded();
    // The published defaults: 1024x1024 at 4 steps.
    assert_eq!((jobs[0].width, jobs[0].height), (1024, 1024));
    assert_eq!(jobs[0].steps, 4);
}

#[tokio::test]
async fn a_named_step_count_overrides_the_server_default() {
    let (state, log) = state_with_mock(false);
    let (status, body) = post_json(
        state,
        "/v1/images",
        json!({"prompt": "a red door", "steps": 8}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(log.recorded()[0].steps, 8);
}

#[tokio::test]
async fn a_step_count_outside_one_to_hundred_is_refused() {
    for steps in [0, 101] {
        let (state, log) = state_with_mock(false);
        let (status, body) = post_json(
            state,
            "/v1/images/generations",
            json!({"prompt": "a red door", "steps": steps}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "steps={steps}: {body}");
        assert_eq!(body["error"]["param"], "steps", "steps={steps}: {body}");
        assert!(log.recorded().is_empty(), "steps={steps}: no job ran");
    }
}

#[tokio::test]
async fn several_images_each_get_seed_plus_index() {
    let (state, log) = state_with_mock(false);
    let (status, body) = post_json(
        state,
        "/v1/images",
        json!({"prompt": "a red door", "n": 3, "seed": 7}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"].as_array().map(Vec::len), Some(3), "{body}");
    let jobs = log.recorded();
    let seeds: Vec<u64> = jobs.iter().map(|job| job.seed.get()).collect();
    assert_eq!(
        seeds,
        vec![7, 8, 9],
        "image i of a seeded request gets seed+i"
    );
}

// ---------------------------------------------------------------------------
// The buffered body.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_buffered_request_that_meets_a_full_queue_is_all_or_nothing() {
    // The ruling this pins: an overload during an n>1 request is a 503
    // for the whole request, not a body with the images that made it -
    // a JSON body with fewer than the n asked for would be a silent
    // contract violation. Made deterministic: one job blocks the worker
    // inside the generator and another fills the one queue slot, so the
    // request's jobs meet the refusal with the worker provably busy; the
    // same `?` in the buffered loop carries a refusal that lands between
    // two images, and it answers the same way - the whole request is the
    // refusal, and no image body is built.
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let generator = Arc::new(GateGenerator {
        started: Mutex::new(Some(started_tx)),
        gate: Mutex::new(release_rx),
    });
    let boxed: Arc<dyn ImageGenerator> = generator;
    let service = ImageService::start(ImageServiceDesc {
        generator: boxed,
        model_id: "mock-flux".to_owned(),
        max_queue: 1,
    })
    .expect("starts image service");
    // Occupy the worker first: the generator blocks inside this job until
    // the test drops the release sender at the end, and the start signal
    // means the worker has taken it off the queue - the slot is empty.
    let blocker = service.clone();
    tokio::spawn(async move {
        let _gone = blocker.generate(filler_job()).await;
    });
    started_rx
        .await
        .expect("the worker is inside the blocking job");
    // Then fill the one queue slot. On this current-thread runtime a yield
    // runs the spawned task past its `try_send` and to its answer await, so
    // the slot is observably full before the request is sent (the
    // `image_service` queue test's trick).
    let filler = service.clone();
    tokio::spawn(async move {
        let _gone = filler.generate(filler_job()).await;
    });
    tokio::task::yield_now().await;
    let state = Arc::new(AppState::without_language_model().with_image(service));

    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door", "n": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "server_overloaded", "{body}");
    assert!(
        body.get("data").is_none(),
        "the answer is the refusal, not the images that made it: {body}"
    );
    // Let the worker finish what it holds: dropping the sender ends every
    // gate wait.
    drop(release_tx);
}

#[tokio::test]
async fn a_buffered_response_round_trips_the_image_and_accounts_usage() {
    let (state, _log) = state_with_mock(false);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door", "size": "512x512"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let data = body["data"].as_array().expect("the data list");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["media_type"], "image/png");
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(data[0]["b64_json"].as_str().expect("b64_json is a string"))
        .expect("b64_json is base64");
    assert_eq!(
        decoded,
        png_bytes(),
        "the bytes come back exactly as generated"
    );
    assert_eq!(body["usage"]["prompt_tokens"], 10);
    assert_eq!(body["usage"]["completion_tokens"], 1024);
    assert_eq!(body["usage"]["total_tokens"], 1034);
    // Anything after 2020: a zero here means the clock was never read.
    assert!(
        body["created"]
            .as_u64()
            .is_some_and(|created| created > 1_577_836_800)
    );
}

// ---------------------------------------------------------------------------
// The SSE stream.
// ---------------------------------------------------------------------------

/// The frames of an SSE body as (event name, data) pairs, in wire order.
/// Frames without an `event:` line - the error frame and the terminator -
/// read with an empty name.
fn sse_frames(text: &str) -> Vec<(&str, &str)> {
    text.split("\n\n")
        .filter(|frame| frame.contains("data:"))
        .map(|frame| {
            let mut name = "";
            let mut data = "";
            for line in frame.lines() {
                if let Some(line) = line.strip_prefix("event:") {
                    name = line.trim();
                } else if let Some(line) = line.strip_prefix("data:") {
                    data = line.trim();
                }
            }
            (name, data)
        })
        .collect()
}

#[tokio::test]
async fn a_streamed_head_arrives_at_admission_not_at_the_first_partial() {
    // The contract the module doc states: the 200 head is flushed by the
    // admission comment, so a client sees the status and the event-stream
    // content type while the first generation is still inside the generator
    // and no partial has been decoded. Before the split this call waited
    // for the first frame, which here means waiting for the release below.
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let generator = Arc::new(GateGenerator {
        started: Mutex::new(Some(started_tx)),
        gate: Mutex::new(release_rx),
    });
    let boxed: Arc<dyn ImageGenerator> = generator;
    let service = ImageService::start(ImageServiceDesc {
        generator: boxed,
        model_id: "mock-flux".to_owned(),
        max_queue: 8,
    })
    .expect("starts image service");
    let state = Arc::new(AppState::without_language_model().with_image(service));
    // Polling the request and the start signal together: the head must
    // arrive while the generator is provably inside the job, gate held.
    let (response, started) = tokio::join!(
        build_router(state).oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"prompt": "a red door", "stream": true}).to_string(),
                ))
                .expect("builds"),
        ),
        started_rx,
    );
    let response = response.expect("the head arrives while the generation is blocked");
    started.expect("the generator started the job");
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/event-stream")),
        "a stream says so"
    );
    // Release the generator and read the stream to its end.
    drop(release_tx);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let text = String::from_utf8(bytes.to_vec()).expect("frames are UTF-8");
    assert!(
        text.contains(": accepted"),
        "the admission comment opens the stream: {text}"
    );
    // The comment is not an event: the wire event shape is unchanged -
    // completed frame, then the terminator.
    let events = sse_frames(&text);
    let names: Vec<&str> = events.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, vec!["image_generation.completed", ""], "{text}");
    assert_eq!(events.last().expect("a last frame").1, "[DONE]");
}

#[tokio::test]
async fn a_streamed_request_frames_partials_then_completed_then_done() {
    let (state, _log) = state_with_mock(true);
    let response = build_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"prompt": "a red door", "stream": true}).to_string(),
                ))
                .expect("builds"),
        )
        .await
        .expect("the router answers");
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/event-stream")),
        "a stream says so"
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let text = String::from_utf8(bytes.to_vec()).expect("frames are UTF-8");

    // The frames in order: one partial per step except the last - the
    // final step's decode is the finished image byte for byte, so its
    // partial frame is skipped - then the completed frame, then the
    // terminator.
    let events = sse_frames(&text);

    let names: Vec<&str> = events.iter().map(|(name, _)| *name).collect();
    let steps = 4; // the server default step count the mock walks
    let partials = steps - 1; // the final step rides the completed frame
    assert_eq!(
        names,
        vec!["image_generation.partial_image"; partials]
            .into_iter()
            .chain(std::iter::once("image_generation.completed"))
            // The terminator rides without an event name of its own.
            .chain(std::iter::once(""))
            .collect::<Vec<_>>(),
        "every step but the last becomes a partial frame, then the answer: {text}"
    );
    assert_eq!(events.last().expect("a last frame").1, "[DONE]");

    // The partial indices increment from zero, and the completed frame carries
    // the final image and the accounting.
    let first_partial: serde_json::Value =
        serde_json::from_str(events[0].1).expect("a partial frame is JSON");
    assert_eq!(first_partial["partial_image_index"], 0);
    assert_eq!(first_partial["type"], "image_generation.partial_image");
    let completed: serde_json::Value =
        serde_json::from_str(events[partials].1).expect("the completed frame is JSON");
    assert_eq!(completed["type"], "image_generation.completed");
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(completed["b64_json"].as_str().expect("b64_json"))
        .expect("b64_json is base64");
    assert_eq!(decoded, png_bytes());
    assert_eq!(completed["usage"]["total_tokens"], 1034);
}

#[tokio::test]
async fn a_stream_that_fails_after_its_first_image_ends_with_an_error_frame() {
    // The contract the module doc states: past the first frame the 200 is
    // spent, so a failure arrives as an error frame and the stream still
    // closes with the terminator rather than truncating silently.
    let generator = Arc::new(FailingSecondGenerator {
        calls: AtomicUsize::new(0),
    });
    let concrete = Arc::clone(&generator);
    let boxed: Arc<dyn ImageGenerator> = concrete;
    let service = ImageService::start(ImageServiceDesc {
        generator: boxed,
        model_id: "mock-flux".to_owned(),
        max_queue: 8,
    })
    .expect("starts image service");
    let state = Arc::new(AppState::without_language_model().with_image(service));
    let response = build_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"prompt": "a red door", "stream": true, "n": 2}).to_string(),
                ))
                .expect("builds"),
        )
        .await
        .expect("the router answers");
    assert_eq!(response.status(), StatusCode::OK, "the 200 is spent");
    // Reading the body to its end is the no-hang assertion: a stream that
    // never closed would leave this waiting, not returning.
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let text = String::from_utf8(bytes.to_vec()).expect("frames are UTF-8");
    let events = sse_frames(&text);

    // The first image's completed frame, then the error frame, then the
    // terminator - and no second completed frame pretending the failure
    // never happened.
    let names: Vec<&str> = events.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        names,
        vec!["image_generation.completed", "", ""],
        "completed, error, terminator: {text}"
    );
    let error: serde_json::Value =
        serde_json::from_str(events[1].1).expect("the error frame is JSON");
    assert_eq!(error["error"]["type"], "server_error", "{error}");
    assert_eq!(error["error"]["code"], "internal_error", "{error}");
    assert_eq!(events.last().expect("a last frame").1, "[DONE]");
    assert_eq!(
        generator.calls.load(Ordering::SeqCst),
        2,
        "the stream stopped at the failing image instead of starting a third"
    );
}

// ---------------------------------------------------------------------------
// Discovery.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_model_listing_names_the_image_model() {
    let (state, _log) = state_with_mock(false);
    let ids = model_ids(state).await;
    assert!(
        ids.iter().any(|id| id == "mock-flux"),
        "the image model is listed: {ids:?}"
    );
}

#[tokio::test]
async fn the_model_listing_names_no_image_model_when_none_is_loaded() {
    // The listing is how a client discovers what to name in `model`, so
    // the mirror of the test above matters just as much: with no generator
    // loaded, no image id may be offered for a request to name.
    let state = Arc::new(AppState::without_language_model());
    let ids = model_ids(state).await;
    assert!(
        !ids.iter().any(|id| id.contains("flux")),
        "no image model is loaded, so none is listed: {ids:?}"
    );
    assert!(
        ids.is_empty(),
        "this state loads no model of any kind: {ids:?}"
    );
}

// ---------------------------------------------------------------------------
// The size grammar, called directly: the table and the clamps.
// ---------------------------------------------------------------------------

#[test]
fn the_exact_shape_passes_through() {
    assert_eq!(
        normalize_size_default("1024x1024").expect("exact"),
        (1024, 1024)
    );
    assert_eq!(
        normalize_size_default("512x512").expect("exact"),
        (512, 512)
    );
    assert_eq!(
        normalize_size_default("2048x256").expect("exact"),
        (2048, 256)
    );
}

#[test]
fn the_aspect_table_names_servable_pixels() {
    let table = [
        ("1:1", (1024u32, 1024u32)),
        ("16:9", (1344, 768)),
        ("9:16", (768, 1344)),
        ("4:3", (1152, 864)),
        ("3:4", (864, 1152)),
        ("2:3", (832, 1248)),
        ("3:2", (1248, 832)),
        ("21:9", (1344, 576)),
    ];
    for (ratio, want) in table {
        assert_eq!(
            normalize_size_default(ratio).expect("a listed ratio"),
            want,
            "{ratio}"
        );
        // Every table entry is a job the generator will accept.
        assert!(want.0.is_multiple_of(16) && (256..=2048).contains(&want.0));
        assert!(want.1.is_multiple_of(16) && (256..=2048).contains(&want.1));
    }
}

#[test]
fn a_nonstandard_size_rounds_to_the_grid_and_clamps_to_the_bounds() {
    assert_eq!(
        normalize_size_default("1000x1000").expect("roundable"),
        (1008, 1008)
    );
    assert_eq!(
        normalize_size_default("996x996").expect("roundable"),
        (992, 992)
    );
    assert_eq!(
        normalize_size_default("100x100").expect("clamped up"),
        (256, 256)
    );
    assert_eq!(
        normalize_size_default("3000x3000").expect("clamped down"),
        (2048, 2048)
    );
}

#[test]
fn a_string_that_is_neither_size_nor_listed_ratio_is_refused() {
    for value in ["banana", "1024", "1024x", "x1024", "13:7", "", "1:1:1"] {
        assert!(
            normalize_size_default(value).is_err(),
            "{value:?} must not name a size"
        );
    }
}

// ---------------------------------------------------------------------------
// Per-model limits: the loaded model, not the server, says what is servable.
// ---------------------------------------------------------------------------

/// A 32-grid model whose schedule is free and whose defaults are its own.
const COARSE: ImageLimits = ImageLimits {
    size_multiple: 32,
    min_side: 256,
    max_side: 2048,
    default_size: (1024, 1024),
    default_steps: 40,
    fixed_steps: None,
};

#[tokio::test]
async fn a_size_off_the_models_grid_is_refused_naming_the_grid() {
    for size in ["1000x1000", "1040x1040"] {
        let (state, log) = state_with_limits(false, COARSE);
        let (status, body) = post_json(
            state,
            "/v1/images/generations",
            json!({"prompt": "a red door", "size": size}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{size}: {body}");
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains("multiples of 32"), "{size}: {body}");
        assert!(log.recorded().is_empty(), "{size}: no job ran");
    }
    let (state, log) = state_with_limits(false, COARSE);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door", "size": "1024x1024"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(log.recorded().len(), 1);
}

#[tokio::test]
async fn a_fixed_schedule_refuses_other_step_counts_and_supplies_its_own() {
    let fixed = ImageLimits {
        fixed_steps: Some(8),
        default_steps: 8,
        ..COARSE
    };
    let (state, log) = state_with_limits(false, fixed);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door", "steps": 4}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "steps", "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("fixed at 8 steps"), "{body}");
    assert!(log.recorded().is_empty());

    let (state, log) = state_with_limits(false, fixed);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(log.recorded()[0].steps, 8);
}

#[tokio::test]
async fn a_request_that_names_nothing_gets_the_models_own_defaults() {
    let (state, log) = state_with_limits(false, COARSE);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let job = &log.recorded()[0];
    assert_eq!((job.width, job.height, job.steps), (1024, 1024, 40));
}
