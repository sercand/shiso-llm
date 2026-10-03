//! The image generation surface, driven through the router with a mocked
//! generator: the statuses a client fixes on its own, the buffered body, the
//! SSE frame order, and the size grammar.
//!
//! The generator is mocked (the `image_service` tests' trick): what is under
//! test is the wire contract - refusals, normalization, base64 round-trip,
//! frame order - not the diffusion math, which needs a device and belongs to
//! `shiso-models`.

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
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageStepEvent,
};
use tower::ServiceExt as _;

/// The encoded bytes the mock hands back: a PNG signature, no pixels.
fn png_bytes() -> Vec<u8> {
    vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
}

/// A generator that records every job it is handed, so a test can read the
/// dimensions and seeds the wire layer actually produced, and can emit partial
/// step images when the request asked to stream.
#[derive(Debug)]
struct RecordingGenerator {
    jobs: Mutex<Vec<ImageGenerationJob>>,
    partials: bool,
}

impl RecordingGenerator {
    fn new(partials: bool) -> Self {
        Self {
            jobs: Mutex::new(Vec::new()),
            partials,
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

/// The state for one test: a server with no language model and the mock
/// generator attached, plus the log the generator writes into.
fn state_with_mock(partials: bool) -> (Arc<AppState>, Arc<RecordingGenerator>) {
    let generator = Arc::new(RecordingGenerator::new(partials));
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

// ---------------------------------------------------------------------------
// The brief's two acceptance tests.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn images_endpoint_returns_503_when_no_image_model() {
    let state = Arc::new(AppState::without_language_model());
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a futuristic city"}),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "image_model_not_loaded", "{body}");
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
async fn a_compression_outside_zero_to_hundred_is_refused() {
    let (state, _log) = state_with_mock(false);
    let (status, body) = post_json(
        state,
        "/v1/images/generations",
        json!({"prompt": "a red door", "output_compression": 101}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "output_compression", "{body}");
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

    // The frames in order: one partial per step (the mock emits bytes with
    // every step), then the completed frame, then the terminator.
    let events: Vec<(&str, &str)> = text
        .split("\n\n")
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
        .collect();

    let names: Vec<&str> = events.iter().map(|(name, _)| *name).collect();
    let steps = 4; // the server default step count the mock walks
    assert_eq!(
        names,
        vec!["image_generation.partial_image"; 4]
            .into_iter()
            .chain(std::iter::once("image_generation.completed"))
            // The terminator rides without an event name of its own.
            .chain(std::iter::once(""))
            .collect::<Vec<_>>(),
        "every step with bytes becomes a partial frame, then the answer: {text}"
    );
    assert_eq!(events.last().expect("a last frame").1, "[DONE]");

    // The partial indices increment from zero, and the completed frame carries
    // the final image and the accounting.
    let first_partial: serde_json::Value =
        serde_json::from_str(events[0].1).expect("a partial frame is JSON");
    assert_eq!(first_partial["partial_image_index"], 0);
    assert_eq!(first_partial["type"], "image_generation.partial_image");
    let completed: serde_json::Value =
        serde_json::from_str(events[steps].1).expect("the completed frame is JSON");
    assert_eq!(completed["type"], "image_generation.completed");
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(completed["b64_json"].as_str().expect("b64_json"))
        .expect("b64_json is base64");
    assert_eq!(decoded, png_bytes());
    assert_eq!(completed["usage"]["total_tokens"], 1034);
}

// ---------------------------------------------------------------------------
// Discovery.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_model_listing_names_the_image_model() {
    let (state, _log) = state_with_mock(false);
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
    let ids: Vec<&str> = body["data"]
        .as_array()
        .expect("a list")
        .iter()
        .filter_map(|entry| entry["id"].as_str())
        .collect();
    assert!(
        ids.contains(&"mock-flux"),
        "the image model is listed: {body}"
    );
}

// ---------------------------------------------------------------------------
// The size grammar, called directly: the table and the clamps.
// ---------------------------------------------------------------------------

#[test]
fn the_exact_shape_passes_through() {
    assert_eq!(normalize_size("1024x1024").expect("exact"), (1024, 1024));
    assert_eq!(normalize_size("512x512").expect("exact"), (512, 512));
    assert_eq!(normalize_size("2048x256").expect("exact"), (2048, 256));
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
            normalize_size(ratio).expect("a listed ratio"),
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
        normalize_size("1000x1000").expect("roundable"),
        (1008, 1008)
    );
    assert_eq!(normalize_size("996x996").expect("roundable"), (992, 992));
    assert_eq!(normalize_size("100x100").expect("clamped up"), (256, 256));
    assert_eq!(
        normalize_size("3000x3000").expect("clamped down"),
        (2048, 2048)
    );
}

#[test]
fn a_string_that_is_neither_size_nor_listed_ratio_is_refused() {
    for value in ["banana", "1024", "1024x", "x1024", "13:7", "", "1:1:1"] {
        assert!(
            normalize_size(value).is_err(),
            "{value:?} must not name a size"
        );
    }
}

#[test]
fn a_seed_newtype_carries_the_wire_value() {
    // The wire hands the job a `Seed`, not a bare u64; the round trip is the
    // contract the generator reads.
    assert_eq!(Seed::new(7).get(), 7);
}
