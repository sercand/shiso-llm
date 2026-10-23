//! The transcription surface and the audio-only server, without a device.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::Value;
use shiso_audio::{
    AudioError, Filterbank, Framing, LogScale, MelScale, MelSpec, PieceVocab, Preemphasis,
    RecognizerStream, Spectrum, SpeechRecognizer, StreamOptions, Window,
};
use shiso_server::audio::{AudioService, AudioServiceDesc};
use shiso_server::router::build_router;
use shiso_server::state::AppState;
use tower::ServiceExt as _;

async fn call(state: Arc<AppState>, request: Request<Body>) -> (StatusCode, Value) {
    let response = build_router(state)
        .oneshot(request)
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn a_generation_route_on_a_server_without_a_language_model_says_so() {
    let state = Arc::new(AppState::without_language_model());
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .expect("builds");
    let (status, body) = call(state, request).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "model_not_loaded", "{body}");
}

#[tokio::test]
async fn a_server_without_a_language_model_lists_none() {
    let state = Arc::new(AppState::without_language_model());
    let request = Request::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .expect("builds");
    let (status, body) = call(state, request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"].as_array().map(Vec::len), Some(0), "{body}");
}

// The HTTP-facing audio cases, moved from the worker module's own tests: they
// need the router, which `shiso-runtime` does not have.

/// The mel front end of the stub recognizer.
fn spec() -> MelSpec {
    MelSpec {
        sample_rate: 16_000,
        n_fft: 512,
        win_length: 400,
        hop_length: 160,
        n_mels: 8,
        window: Window::HannSymmetric,
        framing: Framing::Centred,
        preemphasis: Preemphasis::None,
        spectrum: Spectrum::Power,
        filterbank: Filterbank::Computed {
            scale: MelScale::Slaney,
            slaney_norm: true,
            f_min: 0.0,
            f_max: 8_000.0,
        },
        log_scale: LogScale::Natural { guard: 1e-6 },
    }
}

/// A recognizer that is never asked to recognise: these tests stop at the
/// upload, before any stream opens.
#[derive(Debug)]
struct Stub {
    spec: MelSpec,
    vocab: Arc<PieceVocab>,
}

impl SpeechRecognizer for Stub {
    fn mel_spec(&self) -> &MelSpec {
        &self.spec
    }
    fn vocab(&self) -> &Arc<PieceVocab> {
        &self.vocab
    }
    fn frame_secs(&self) -> f64 {
        0.08
    }
    fn languages(&self) -> Vec<String> {
        vec!["en-US".into()]
    }
    fn lookaheads(&self) -> &[usize] {
        &[0]
    }
    fn is_special(&self, id: u32) -> bool {
        id == 0
    }
    fn open_stream(
        &self,
        _options: &StreamOptions,
    ) -> Result<Box<dyn RecognizerStream>, AudioError> {
        Err(AudioError::Empty)
    }
}

fn service_with_upload(max_upload_bytes: usize) -> AudioService {
    let vocab = Arc::new(
        PieceVocab::from_pieces(vec![
            "<blank>".into(),
            "\u{2581}a".into(),
            "\u{2581}b".into(),
        ])
        .expect("pieces"),
    );
    let (service, _join) = AudioService::start(AudioServiceDesc {
        recognizer: Arc::new(Stub {
            spec: spec(),
            vocab,
        }),
        diarizer: None,
        asr_id: "scripted".into(),
        lookahead: 0,
        max_streams: 1,
        queue: 1,
        limits: shiso_server::tokenizer::message_preprocess::DEFAULT_AUDIO_LIMITS,
        decode: shiso_audio::AudioDecodePolicy::default(),
        max_upload_bytes,
        offline_min_secs: 0,
        skip_silence: false,
        max_context_ids: 3,
    })
    .expect("the thread starts");
    service
}

#[tokio::test]
async fn an_audio_only_server_lists_its_recognizer() {
    let state = AppState::without_language_model().with_audio(service_with_upload(1024));
    let listing = shiso_server::api::misc_handlers::model_listing(&state);
    let ids: Vec<&str> = listing.data.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, vec!["scripted"]);
}

/// A transcription form carrying `file_bytes` bytes that are not audio.
fn upload(file_bytes: usize) -> Request<Body> {
    let boundary = "shisoboundary";
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nscripted\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.bin\"\r\n\
         Content-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    body.extend(std::iter::repeat_n(7u8, file_bytes));
    body.extend(format!("\r\n--{boundary}--\r\n").into_bytes());
    Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .expect("builds")
}

async fn status_and_code(service: AudioService, request: Request<Body>) -> (StatusCode, Value) {
    call(
        Arc::new(AppState::without_language_model().with_audio(service)),
        request,
    )
    .await
}

#[tokio::test]
async fn an_upload_past_the_json_body_limit_is_read_as_audio() {
    // 40 MiB: past the 32 MiB every other route takes, inside the audio limit.
    let (status, body) = status_and_code(
        service_with_upload(64 * 1024 * 1024),
        upload(40 * 1024 * 1024),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["code"], "invalid_audio",
        "it was read: {body}"
    );
}

#[tokio::test]
async fn an_upload_past_the_audio_limit_is_refused_as_too_large() {
    let (status, body) =
        status_and_code(service_with_upload(1024 * 1024), upload(2 * 1024 * 1024)).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(body["error"]["code"], "request_too_large", "{body}");
}
