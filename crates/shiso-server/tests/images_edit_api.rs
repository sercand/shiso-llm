//! `POST /v1/images/edits`, driven through the router with a mocked generator:
//! the references reach the generator intact, and every refusal happens
//! before the generator is asked to do anything.
//!
//! The generator is mocked for the reason the `images_api` tests give: what is
//! under test is the wire contract, not the diffusion math, which needs a
//! device.

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use serde_json::json;
use shiso_server::image::{ImageService, ImageServiceDesc};
use shiso_server::router::build_router;
use shiso_server::state::AppState;
use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageLimits, ImageStepEvent,
    InputReferenceImage,
};
use tower::ServiceExt as _;

const LIMITS: ImageLimits = ImageLimits {
    size_multiple: 16,
    min_side: 256,
    max_side: 2048,
    default_size: (1024, 1024),
    default_steps: 4,
    fixed_steps: None,
};

/// The size the mock derives from any references: not the text default, so a
/// test can tell which of the two the handler used.
const REFERENCE_SIZE: (u32, u32) = (640, 384);

/// Records every job; takes `max_references` references.
#[derive(Debug)]
struct EditMock {
    max_references: usize,
    jobs: Mutex<Vec<ImageGenerationJob>>,
}

impl ImageGenerator for EditMock {
    fn model_id(&self) -> &'static str {
        "mock-editor"
    }

    fn limits(&self) -> ImageLimits {
        LIMITS
    }

    fn max_references(&self) -> usize {
        self.max_references
    }

    fn default_size_for_references(
        &self,
        _references: &[InputReferenceImage],
    ) -> Result<(u32, u32), ImageGenError> {
        Ok(REFERENCE_SIZE)
    }

    fn generate(
        &self,
        job: &ImageGenerationJob,
        _progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        self.jobs
            .lock()
            .expect("the log lock is never poisoned")
            .push(job.clone());
        Ok(GeneratedImage {
            bytes: vec![0x89, b'P', b'N', b'G'],
            media_type: "image/png",
            width: job.width,
            height: job.height,
            prompt_tokens: 1,
            completion_tokens: 1,
        })
    }
}

fn state(max_references: usize) -> (Arc<AppState>, Arc<EditMock>) {
    let mock = Arc::new(EditMock {
        max_references,
        jobs: Mutex::new(Vec::new()),
    });
    let generator: Arc<dyn ImageGenerator> = mock.clone();
    let service = ImageService::start(ImageServiceDesc {
        generator,
        model_id: "mock-editor".to_owned(),
        max_queue: 4,
        profile: None,
    })
    .expect("starts image service");
    (
        Arc::new(AppState::without_language_model().with_image(service)),
        mock,
    )
}

/// A multipart body of `(name, bytes)` parts.
fn multipart(parts: &[(&str, &[u8])]) -> (String, Vec<u8>) {
    let boundary = "shisoBoundary7f3a";
    let mut body = Vec::new();
    for (name, bytes) in parts {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        let file = if name.starts_with("image") {
            "; filename=\"ref.png\"\r\nContent-Type: image/png"
        } else {
            ""
        };
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"{file}\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn send(
    state: Arc<AppState>,
    uri: &str,
    content_type: &str,
    body: Vec<u8>,
) -> (StatusCode, serde_json::Value) {
    let response = build_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", content_type)
                .body(Body::from(body))
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

/// `count` distinguishable "pictures": the mock never decodes them.
fn pictures(count: u8) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| vec![i, 0xAA, 0x55, i.wrapping_mul(7), 0, 1, 2])
        .collect()
}

async fn edit(
    state: Arc<AppState>,
    pics: &[Vec<u8>],
    name: &str,
    extra: &[(&str, &str)],
) -> (StatusCode, serde_json::Value) {
    let mut parts: Vec<(&str, &[u8])> = pics.iter().map(|p| (name, p.as_slice())).collect();
    for (key, value) in extra {
        parts.push((key, value.as_bytes()));
    }
    let (content_type, body) = multipart(&parts);
    send(state, "/v1/images/edits", &content_type, body).await
}

#[tokio::test]
async fn one_reference_reaches_the_generator_intact_at_the_reference_size() {
    let (state, mock) = state(10);
    let pics = pictures(1);
    let (status, body) = edit(state, &pics, "image", &[("prompt", "a hat"), ("seed", "5")]).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["data"][0]["b64_json"].is_string());
    let jobs = mock.jobs.lock().expect("lock").clone();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].input_references.len(), 1);
    assert_eq!(jobs[0].input_references[0].bytes, pics[0]);
    assert_eq!((jobs[0].width, jobs[0].height), REFERENCE_SIZE);
    assert_eq!(jobs[0].prompt, "a hat");
    assert_eq!(jobs[0].seed.get(), 5);
}

#[tokio::test]
async fn three_references_arrive_in_order_under_the_bracket_spelling() {
    let (state, mock) = state(10);
    let pics = pictures(3);
    let (status, body) = edit(
        state,
        &pics,
        "image[]",
        &[("prompt", "merge"), ("size", "512x512")],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let jobs = mock.jobs.lock().expect("lock").clone();
    let got: Vec<Vec<u8>> = jobs[0]
        .input_references
        .iter()
        .map(|r| r.bytes.clone())
        .collect();
    assert_eq!(got, pics);
    // A named size wins over the references' size.
    assert_eq!((jobs[0].width, jobs[0].height), (512, 512));
}

#[tokio::test]
async fn the_json_shape_with_data_urls_works() {
    let (state, mock) = state(10);
    let pics = pictures(2);
    let images: Vec<_> = pics
        .iter()
        .map(|p| {
            json!({"image_url": format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(p)
            )})
        })
        .collect();
    let body = json!({"prompt": "x", "images": images})
        .to_string()
        .into_bytes();
    let (status, answer) = send(state, "/images/edits", "application/json", body).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let jobs = mock.jobs.lock().expect("lock").clone();
    assert_eq!(jobs[0].input_references.len(), 2);
    assert_eq!(jobs[0].input_references[1].bytes, pics[1]);
}

#[tokio::test]
async fn eleven_references_are_refused_naming_the_limit_without_generating() {
    let (state, mock) = state(10);
    let (status, body) = edit(state, &pictures(11), "image", &[("prompt", "x")]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = body["error"]["message"].as_str().expect("message");
    assert!(message.contains("at most 10"), "{message}");
    assert!(mock.jobs.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn an_n_other_than_one_is_refused() {
    let (state, mock) = state(10);
    let (status, body) = edit(state, &pictures(1), "image", &[("prompt", "x"), ("n", "2")]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "n");
    assert!(mock.jobs.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn a_model_without_reference_support_is_refused_by_name() {
    let (state, mock) = state(0);
    let (status, body) = edit(state, &pictures(1), "image", &[("prompt", "x")]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = body["error"]["message"].as_str().expect("message");
    assert!(
        message.contains("mock-editor") && message.contains("does not take reference images"),
        "{message}"
    );
    assert!(mock.jobs.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn a_missing_prompt_a_missing_image_and_a_mask_are_refused() {
    let (state, _mock) = state(10);
    let (status, body) = edit(state.clone(), &pictures(1), "image", &[]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "prompt");
    let (status, body) = edit(state.clone(), &[], "image", &[("prompt", "x")]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "image");
    let (status, body) = edit(state, &pictures(1), "mask", &[("prompt", "x")]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn a_url_response_format_is_refused_as_on_the_generations_route() {
    let (state, _mock) = state(10);
    let (status, body) = edit(
        state,
        &pictures(1),
        "image",
        &[("prompt", "x"), ("response_format", "url")],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "response_format");
}

#[tokio::test]
async fn generations_stay_text_only_and_at_the_text_default_size() {
    let (state, mock) = state(10);
    let body = json!({"prompt": "a fox"}).to_string().into_bytes();
    let (status, answer) = send(state, "/v1/images/generations", "application/json", body).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let jobs = mock.jobs.lock().expect("lock").clone();
    assert!(jobs[0].input_references.is_empty());
    assert_eq!((jobs[0].width, jobs[0].height), (1024, 1024));
}
