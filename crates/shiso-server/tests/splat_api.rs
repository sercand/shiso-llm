//! The 3D generation surface, driven through the router with a mocked
//! generator: the statuses a client fixes on its own, the buffered body, the
//! SSE frame order, and the defaults the wire layer fills in.
//!
//! The generator is mocked (the `images_api` tests' trick): what is under test
//! is the wire contract - refusals, base64 round-trip, frame order - not the
//! network, which needs a device and belongs to `shiso-models`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use serde_json::json;
use shiso_core::ids::Seed;
use shiso_server::router::build_router;
use shiso_server::splat::{SplatService, SplatServiceDesc};
use shiso_server::state::AppState;
use shiso_vision::splat::{
    GeneratedSplat, SplatGenError, SplatGenerator, SplatJob, SplatLimits, SplatStepEvent,
};
use tower::ServiceExt as _;

/// The encoded bytes the mock hands back: a `.ply` signature, no vertices.
fn ply_bytes() -> Vec<u8> {
    b"ply\nformat binary_little_endian 1.0\n".to_vec()
}

/// A generator that records every job it is handed, so a test can read the
/// defaults the wire layer actually produced, and emits one step event per
/// step so a stream has something to forward.
#[derive(Debug)]
struct RecordingGenerator {
    jobs: Mutex<Vec<SplatJob>>,
    calls: AtomicUsize,
}

impl RecordingGenerator {
    fn new() -> Self {
        Self {
            jobs: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
        }
    }

    fn recorded(&self) -> Vec<SplatJob> {
        self.jobs
            .lock()
            .expect("the log lock is never poisoned")
            .clone()
    }
}

impl SplatGenerator for RecordingGenerator {
    fn model_id(&self) -> &'static str {
        "mock-splat"
    }

    fn limits(&self) -> SplatLimits {
        SplatLimits::default()
    }

    fn generate(
        &self,
        job: &SplatJob,
        progress: Option<&dyn Fn(SplatStepEvent)>,
    ) -> Result<GeneratedSplat, SplatGenError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.jobs
            .lock()
            .expect("the log lock is never poisoned")
            .push(job.clone());
        if let Some(progress) = progress {
            for step in 0..job.steps {
                progress(SplatStepEvent {
                    step,
                    total_steps: job.steps,
                });
            }
        }
        Ok(GeneratedSplat {
            bytes: ply_bytes(),
            media_type: "model/ply",
            num_gaussians: job.num_gaussians,
            completion_tokens: 12_294 * job.steps,
        })
    }
}

/// A state with the mock attached.
fn state_with_mock() -> (Arc<AppState>, Arc<RecordingGenerator>) {
    let generator = Arc::new(RecordingGenerator::new());
    let boxed: Arc<dyn SplatGenerator> = Arc::<RecordingGenerator>::clone(&generator);
    let service = SplatService::start(SplatServiceDesc {
        generator: boxed,
        model_id: "mock-splat".to_owned(),
        max_queue: 8,
        profile: None,
    })
    .expect("starts splat service");
    let state = Arc::new(AppState::without_language_model().with_splat(service));
    (state, generator)
}

/// POSTs one JSON body at the route and reads the whole answer.
async fn post_json(
    state: Arc<AppState>,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let response = build_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/3d/generations")
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

/// A minimal valid body: one base64 picture.
fn body() -> serde_json::Value {
    json!({"image": "aGVsbG8="})
}

#[tokio::test]
async fn a_blocking_request_answers_with_the_encoded_splat() {
    let (state, _generator) = state_with_mock();
    let (status, body) = post_json(state, body()).await;
    assert_eq!(status, StatusCode::OK);
    let encoded = body["data"][0]["b64_ply"].as_str().expect("a b64_ply");
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .expect("decodes"),
        ply_bytes()
    );
    assert_eq!(body["usage"]["prompt_tokens"], 0);
    assert_eq!(body["usage"]["total_tokens"], 12_294 * 50);
    assert!(body["created"].as_u64().is_some_and(|value| value > 0));
}

#[tokio::test]
async fn a_bad_knob_is_a_400_even_without_a_model() {
    // A body this server will never act on is the caller's fault whether or
    // not a splat model is loaded; answering 503 first would send the caller
    // to fix a flag only for the same request to start failing 400.
    let state = Arc::new(AppState::without_language_model());
    let (status, body) = post_json(state, json!({"image": "aGVsbG8=", "steps": 0})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "steps");
}

#[tokio::test]
async fn a_server_without_a_splat_model_answers_503() {
    let state = Arc::new(AppState::without_language_model());
    let (status, body) = post_json(state, body()).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "splat_model_not_loaded");
}

#[tokio::test]
async fn an_unknown_mode_is_a_400_naming_the_field() {
    let (state, _generator) = state_with_mock();
    let (status, body) = post_json(state, json!({"image": "aGVsbG8=", "mode": "cartoon"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "mode");
}

#[tokio::test]
async fn a_picture_that_is_not_base64_is_a_400_naming_the_image() {
    let (state, _generator) = state_with_mock();
    let (status, body) = post_json(state, json!({"image": "not base64!!"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "image");
}

#[tokio::test]
async fn the_omitted_knobs_become_the_model_defaults() {
    let (state, generator) = state_with_mock();
    let (status, _body) = post_json(state, json!({"image": "aGVsbG8=", "seed": 7})).await;
    assert_eq!(status, StatusCode::OK);
    let job = generator.recorded().pop().expect("one job");
    assert_eq!(job.seed, Seed::new(7));
    assert_eq!((job.steps, job.num_gaussians), (50, 262_144));
    assert!((job.cfg_strength - 3.0).abs() < f32::EPSILON);
    assert!(
        (job.crop_margin - 1.2).abs() < f32::EPSILON,
        "synthetic mode"
    );
}

#[tokio::test]
async fn real_mode_moves_the_crop_margin_and_an_explicit_one_wins() {
    let (state, generator) = state_with_mock();
    let (status, _body) = post_json(
        state,
        json!({"image": "aGVsbG8=", "mode": "real", "seed": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!((generator.recorded().pop().expect("one job").crop_margin - 1.7).abs() < f32::EPSILON);

    let (state, generator) = state_with_mock();
    let (status, _body) = post_json(
        state,
        json!({"image": "aGVsbG8=", "mode": "real", "crop_margin": 1.4, "seed": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!((generator.recorded().pop().expect("one job").crop_margin - 1.4).abs() < f32::EPSILON);
}

/// A generator that reports it has started and then waits until the test
/// releases it: how a job is caught occupying the worker while the queue
/// behind it fills.
#[derive(Debug)]
struct BlockingGenerator {
    started: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    gate: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl SplatGenerator for BlockingGenerator {
    fn model_id(&self) -> &'static str {
        "blocking"
    }

    fn limits(&self) -> SplatLimits {
        SplatLimits::default()
    }

    fn generate(
        &self,
        _job: &SplatJob,
        _progress: Option<&dyn Fn(SplatStepEvent)>,
    ) -> Result<GeneratedSplat, SplatGenError> {
        if let Some(started) = self.started.lock().expect("lock").take() {
            let _sent = started.send(());
        }
        let _released = self.gate.lock().expect("lock").recv();
        Ok(GeneratedSplat {
            bytes: ply_bytes(),
            media_type: "model/ply",
            num_gaussians: 262_144,
            completion_tokens: 1,
        })
    }
}

#[tokio::test]
async fn a_streamed_request_refused_by_a_full_queue_answers_503() {
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (gate_tx, gate_rx) = std::sync::mpsc::channel();
    let generator = Arc::new(BlockingGenerator {
        started: Mutex::new(Some(started_tx)),
        gate: Mutex::new(gate_rx),
    });
    let boxed: Arc<dyn SplatGenerator> = Arc::<BlockingGenerator>::clone(&generator);
    let service = SplatService::start(SplatServiceDesc {
        generator: boxed,
        model_id: "blocking".to_owned(),
        max_queue: 1,
        profile: None,
    })
    .expect("starts splat service");
    let state = Arc::new(AppState::without_language_model().with_splat(service));

    // The first request occupies the worker; the second fills the queue.
    let first = tokio::spawn({
        let state = Arc::clone(&state);
        async move { post_json(state, body()).await }
    });
    started_rx.await.expect("the first job starts");
    let second = tokio::spawn({
        let state = Arc::clone(&state);
        async move { post_json(state, body()).await }
    });
    // Give the second request time to reach the queue before the third is
    // sent; it is a queue slot, not a device call, so this is short.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // A streamed request that the queue refuses must answer with a status
    // code, not a 200 and an error frame.
    let (status, body) = post_json(
        Arc::clone(&state),
        json!({"image": "aGVsbG8=", "stream": true}),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "server_overloaded");

    // Dropping the sender releases every blocked job: the queued one runs
    // after the first, and a second `send` would leave it waiting for a
    // message that never comes.
    drop(gate_tx);
    let _ = first.await.expect("the first request finishes");
    let _ = second.await.expect("the second request finishes");
}

#[tokio::test]
async fn a_streamed_request_sends_admitted_progress_completed_and_done() {
    let (state, _generator) = state_with_mock();
    let response = build_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/3d/generations")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"image": "aGVsbG8=", "stream": true, "steps": 2, "seed": 3}).to_string(),
                ))
                .expect("builds"),
        )
        .await
        .expect("the router answers");
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let text = String::from_utf8(body.to_vec()).expect("SSE is text");
    let types: Vec<String> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .filter_map(|value| value["type"].as_str().map(ToOwned::to_owned))
        .collect();
    assert_eq!(
        types,
        vec![
            "splat_generation.admitted",
            "splat_generation.progress",
            "splat_generation.progress",
            "splat_generation.completed"
        ]
    );
    assert!(text.contains("data: [DONE]"), "the stream terminates");
    // The completed frame carries the file and its accounting.
    let completed = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .find(|value| value["type"] == "splat_generation.completed")
        .expect("a completed frame");
    assert_eq!(completed["usage"]["completion_tokens"], 12_294 * 2);
    assert_eq!(
        completed["b64_ply"],
        base64::engine::general_purpose::STANDARD.encode(ply_bytes())
    );
}
