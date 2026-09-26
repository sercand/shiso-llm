//! `POST /v1/audio/transcriptions` end to end: the binary's own audio boot,
//! the router, real checkpoints and a GPU device (Metal or CUDA).
//!
//! Needs `SHISO_TEST_MODEL_ASR` and `SHISO_TEST_MODEL_DIAR` (the model roots)
//! and the committed audio clips; says why on stderr when it cannot run.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::Value;
use shiso_server::cli::{Cli, Command};
use shiso_server::router::build_router;
use shiso_server::state::AppState;
use tower::ServiceExt as _;

#[expect(clippy::print_stderr, reason = "a skipped case says why on stderr")]
fn say(line: &str) {
    eprintln!("{line}");
}

const BOUNDARY: &str = "shiso-test-boundary";

fn clip(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../test-vectors/references/audio/clips/{name}.wav"
    ));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A multipart body of `fields` and the audio file.
fn form(audio: &[u8], fields: &[(&str, &str)]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n\
             Content-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(audio);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    body
}

/// The audio-only state, booted as the binary boots it.
fn booted() -> Option<Arc<AppState>> {
    let (Some(asr), Some(diar)) = (
        std::env::var_os("SHISO_TEST_MODEL_ASR").map(PathBuf::from),
        std::env::var_os("SHISO_TEST_MODEL_DIAR").map(PathBuf::from),
    ) else {
        say("SKIP: SHISO_TEST_MODEL_ASR and SHISO_TEST_MODEL_DIAR are needed");
        return None;
    };
    // On macOS the device is Metal and may be absent; on Linux the boot below
    // opens CUDA device 0 and says so if it cannot.
    #[cfg(target_os = "macos")]
    if !shiso_metal::metal_available() {
        say("SKIP: no Metal device");
        return None;
    }
    let asr = asr.join("nvidia").display().to_string();
    let diar = diar.join("nvidia").display().to_string();
    let line = [
        "shiso",
        "serve",
        "--asr-model",
        &asr,
        "--asr-id",
        "nemotron-asr",
        "--diarization-model",
        &diar,
    ];
    let cli = <Cli as clap::Parser>::try_parse_from(line).expect("the command line parses");
    let Command::Serve(args) = cli.command;
    args.validate().expect("servable");
    let audio = shiso_server::boot_audio(&args)
        .expect("the speech models load")
        .expect("an ASR was named");
    Some(Arc::new(
        AppState::without_language_model().with_audio(audio),
    ))
}

async fn post(state: &Arc<AppState>, body: Vec<u8>) -> (StatusCode, String) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .expect("builds");
    let response = build_router(Arc::clone(state))
        .oneshot(request)
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    (status, String::from_utf8(bytes.to_vec()).expect("text"))
}

/// The reference transcript of `speech-en` at lookahead 13, language tags
/// dropped as the detokenizer drops them.
fn reference_text() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/audio/asr-speech-en-r13-auto.json");
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json");
    let text = meta["text"].as_str().expect("text");
    let mut out = String::new();
    let mut inside = false;
    for c in text.chars() {
        match c {
            '<' => inside = true,
            '>' => inside = false,
            _ if !inside => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the local speech checkpoints and a GPU device"]
async fn transcriptions_end_to_end() {
    let Some(state) = booted() else { return };
    let speech = clip("speech-en");

    let (status, text) = post(&state, form(&speech, &[("model", "nemotron-asr")])).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let json: Value = serde_json::from_str(&text).expect("json");
    let transcript = json["text"].as_str().expect("text").to_owned();
    say(&format!("json: {transcript:?}"));
    assert_eq!(
        transcript.split_whitespace().collect::<Vec<_>>().join(" "),
        reference_text()
    );
    assert_eq!(json["usage"]["seconds"], 30);

    let (status, stream) = post(
        &state,
        form(&speech, &[("model", "nemotron-asr"), ("stream", "true")]),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{stream}");
    let frames: Vec<Value> = stream
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).expect("a json frame"))
        .collect();
    let deltas: String = frames
        .iter()
        .filter(|f| f["type"] == "transcript.text.delta")
        .filter_map(|f| f["delta"].as_str())
        .collect();
    let last = frames.last().expect("frames");
    assert_eq!(last["type"], "transcript.text.done", "{last}");
    assert_eq!(last["text"], transcript.as_str());
    assert_eq!(deltas.trim_end(), transcript);
    assert!(!transcript.ends_with(' '), "{transcript:?}");

    let (status, srt) = post(
        &state,
        form(
            &speech,
            &[("model", "nemotron-asr"), ("response_format", "srt")],
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{srt}");
    assert!(
        srt.starts_with("1\n00:00:0") && srt.contains(" --> "),
        "{srt}"
    );

    let (status, verbose) = post(
        &state,
        form(
            &speech,
            &[
                ("model", "nemotron-asr"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
            ],
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{verbose}");
    let verbose: Value = serde_json::from_str(&verbose).expect("json");
    let words = verbose["words"].as_array().expect("words");
    assert!(words.len() > 50, "{} words", words.len());
    assert!(
        words
            .windows(2)
            .all(|w| w[0]["start"].as_f64() <= w[1]["start"].as_f64())
    );

    let (status, diarized) = post(
        &state,
        form(
            &clip("two-speakers"),
            &[
                ("model", "nemotron-asr"),
                ("response_format", "diarized_json"),
            ],
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{diarized}");
    let diarized: Value = serde_json::from_str(&diarized).expect("json");
    let segments = diarized["segments"].as_array().expect("segments");
    let mut speakers: Vec<&str> = segments
        .iter()
        .filter_map(|s| s["speaker"].as_str())
        .collect();
    say(&format!(
        "diarized: {} segments, speakers in order {speakers:?}",
        segments.len()
    ));
    assert_eq!(speakers.first(), Some(&"A"), "the first voice is A");
    speakers.sort_unstable();
    speakers.dedup();
    assert_eq!(speakers, vec!["A", "B"]);

    let (status, wrong) = post(&state, form(&speech, &[("model", "whisper-1")])).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{wrong}");
}
