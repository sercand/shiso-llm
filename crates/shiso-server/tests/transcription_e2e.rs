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
    let Command::Serve(args) = cli.command else {
        panic!("expected Command::Serve");
    };
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

/// The chunking recognizer's state: the `Q8_0` text GGUF (its mmproj found
/// beside it) and the MLX diarizer, booted as the binary boots them with
/// `extra` flags. Needs `SHISO_TEST_MODEL_QWEN3ASR` and
/// `SHISO_TEST_MODEL_DIAR`.
#[cfg(target_os = "macos")]
fn booted_chunking(extra: &[&str]) -> Option<(Arc<AppState>, PathBuf)> {
    let (Some(asr), Some(diar)) = (
        std::env::var_os("SHISO_TEST_MODEL_QWEN3ASR").map(PathBuf::from),
        std::env::var_os("SHISO_TEST_MODEL_DIAR").map(PathBuf::from),
    ) else {
        say("SKIP: SHISO_TEST_MODEL_QWEN3ASR and SHISO_TEST_MODEL_DIAR are needed");
        return None;
    };
    if !shiso_metal::metal_available() {
        say("SKIP: no Metal device");
        return None;
    }
    let asr = asr.join("gguf/Qwen3-ASR-1.7B-Q8_0.gguf");
    let diar = diar.join("mlx");
    for path in [&asr, &diar] {
        if !path.exists() {
            say(&format!("SKIP: {} is absent", path.display()));
            return None;
        }
    }
    let (asr_arg, diar_arg) = (asr.display().to_string(), diar.display().to_string());
    let mut line = vec![
        "shiso",
        "serve",
        "--asr-model",
        &asr_arg,
        "--asr-id",
        "qwen3",
        "--diarization-model",
        &diar_arg,
    ];
    line.extend_from_slice(extra);
    let cli = <Cli as clap::Parser>::try_parse_from(line).expect("the command line parses");
    let Command::Serve(args) = cli.command else {
        panic!("expected Command::Serve");
    };
    args.validate().expect("servable");
    let audio = shiso_server::boot_audio(&args)
        .expect("the speech models load")
        .expect("an ASR was named");
    Some((
        Arc::new(AppState::without_language_model().with_audio(audio)),
        diar,
    ))
}

/// The diarizer's activity over `name` and the recognizer's mel of it, as
/// the worker computes them.
#[cfg(target_os = "macos")]
fn activity_and_mel(
    diar: &Path,
    name: &str,
) -> (
    shiso_audio::transcript::SpeakerActivity,
    shiso_audio::MelFrames,
) {
    use shiso_audio::{DecodeLimits, LogScale, MelFrontend, decode_audio};

    let limits = DecodeLimits::new(std::num::NonZeroU32::new(600).expect("non-zero"));
    let pcm = decode_audio(&clip(name), &limits, None).expect("decodes");
    let backend: Arc<dyn shiso_core::Backend> =
        shiso_metal::MetalBackend::new().expect("the Metal library compiles");
    let diarizer = shiso_models::audio_registry::open_diarizer(diar, &backend).expect("loads");
    let dmel = MelFrontend::new(diarizer.mel_spec().clone())
        .expect("a frontend")
        .compute(pcm.samples());
    let activity = diarizer.diarize(&dmel).expect("diarizes");
    let mut spec = shiso_models::qwen3_asr::config::mel_spec();
    spec.log_scale = LogScale::Log10;
    let mel = MelFrontend::new(spec)
        .expect("a frontend")
        .compute(pcm.samples());
    (activity, mel)
}

/// For each chunk the frames `bounds` bound, the speaker who holds most of
/// its diarizer frames.
#[cfg(target_os = "macos")]
fn owners(
    activity: &shiso_audio::transcript::SpeakerActivity,
    bounds: &[u32],
) -> Vec<Option<usize>> {
    use shiso_audio::chunks;

    let fs = activity.frame_secs();
    bounds
        .windows(2)
        .map(|w| {
            let mut count = vec![0usize; activity.speakers()];
            for t in 0..activity.frames() {
                // The diarizer frame's middle, in mel frames.
                let mel_t = (f64::from(u32::try_from(t).expect("a frame fits")) + 0.5) * fs / 0.01;
                if mel_t < f64::from(w[0]) || mel_t >= f64::from(w[1]) {
                    continue;
                }
                let mut best = (None, chunks::DOMINANT_ABOVE);
                for s in 0..activity.speakers() {
                    if activity.prob(t, s) > best.1 {
                        best = (Some(s), activity.prob(t, s));
                    }
                }
                if let Some(s) = best.0 {
                    count[s] += 1;
                }
            }
            let (who, n) = count
                .iter()
                .enumerate()
                .max_by_key(|(s, n)| (**n, std::cmp::Reverse(*s)))
                .expect("a speaker");
            (*n > 0).then_some(who)
        })
        .collect()
}

/// The cuts the worker plans for `name` - the diarizer's turns over the
/// recognizer's frames, or with `offline` the whole-clip plan with its
/// silences skipped - and, for each chunk they bound, the speaker who holds
/// most of its diarizer frames. Skipped spans hold no words, so they own no
/// turn.
#[cfg(target_os = "macos")]
fn turn_plan(diar: &Path, name: &str, offline: bool) -> (Vec<u32>, Vec<Option<usize>>) {
    use shiso_audio::chunks;

    let (activity, mel) = activity_and_mel(diar, name);
    let frame = |f: usize| u32::try_from(f).expect("a frame fits");
    let (cuts, skipped): (Vec<u32>, Vec<bool>) = if offline {
        let spans = chunks::plan_offline(
            &mel,
            Some(&activity),
            0.01,
            &chunks::OfflineRule::for_10ms_mel(true),
        )
        .expect("plans");
        let skipped = spans.iter().map(|s| !s.transcribe).collect();
        let cuts = spans[..spans.len() - 1]
            .iter()
            .map(|s| frame(s.end))
            .collect();
        (cuts, skipped)
    } else {
        let cuts: Vec<u32> = chunks::turn_cuts(&activity, &mel, 0.01)
            .into_iter()
            .map(frame)
            .collect();
        let skipped = vec![false; cuts.len() + 1];
        (cuts, skipped)
    };
    let mut bounds = vec![0];
    bounds.extend(&cuts);
    bounds.push(frame(mel.valid()));
    let owners = owners(&activity, &bounds)
        .into_iter()
        .zip(skipped)
        .map(|(owner, skipped)| owner.filter(|_| !skipped))
        .collect();
    (cuts, owners)
}

/// Every response format on one clip answers with text.
#[cfg(target_os = "macos")]
async fn every_format(state: &Arc<AppState>, name: &str) {
    let audio = clip(name);
    for format in [
        "json",
        "text",
        "verbose_json",
        "srt",
        "vtt",
        "diarized_json",
    ] {
        let started = std::time::Instant::now();
        let (status, body) = post(
            state,
            form(&audio, &[("model", "qwen3"), ("response_format", format)]),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{name} {format}: {body}");
        let text = match format {
            "json" | "verbose_json" | "diarized_json" => {
                let json: Value = serde_json::from_str(&body).expect("json");
                if format == "verbose_json" {
                    assert!(json["language"].is_string(), "{name}: {json}");
                    let words = json["segments"].as_array().expect("segments");
                    assert!(!words.is_empty(), "{name}: no segments");
                }
                json["text"].as_str().expect("text").to_owned()
            }
            _ => body.clone(),
        };
        say(&format!(
            "{name} {format} in {:.2} s: {} chars",
            started.elapsed().as_secs_f64(),
            text.len()
        ));
        assert!(
            text.chars().any(char::is_alphanumeric),
            "{name} {format}: {body}"
        );
    }
    let (status, stream) = post(
        state,
        form(&audio, &[("model", "qwen3"), ("stream", "true")]),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{stream}");
    let last = stream
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str::<Value>(d).expect("a json frame"))
        .next_back()
        .expect("frames");
    assert_eq!(last["type"], "transcript.text.done", "{last}");
}

/// A diarized transcript of `name` changes speaker only on a turn cut, and
/// exactly as often as the turns do.
#[cfg(target_os = "macos")]
async fn speakers_follow_turns(state: &Arc<AppState>, diar: &Path, name: &str, offline: bool) {
    let (cuts, owners) = turn_plan(diar, name, offline);
    let (status, body) = post(
        state,
        form(
            &clip(name),
            &[("model", "qwen3"), ("response_format", "diarized_json")],
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let json: Value = serde_json::from_str(&body).expect("json");
    let segments = json["segments"].as_array().expect("segments");
    assert!(!segments.is_empty(), "{name}: no segments");
    let secs = |v: &Value, key: &str| v[key].as_f64().expect("a time");
    let mut changes = 0;
    for pair in segments.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if a["speaker"] == b["speaker"] {
            continue;
        }
        changes += 1;
        let start = secs(b, "start") * 100.0;
        let at = cuts
            .iter()
            .copied()
            .find(|&c| (f64::from(c) - start).abs() < 0.5);
        let Some(at) = at else {
            panic!("{name}: a speaker change at frame {start} is not on a cut {cuts:?}");
        };
        // The turn's segment runs to its cut, not to its last word's start:
        // the same cut, or, across a skipped silence, the silence's start.
        let end = secs(a, "end") * 100.0;
        let on_cut = if offline {
            cuts.iter().any(|&c| (f64::from(c) - end).abs() < 0.5)
        } else {
            (end - f64::from(at)).abs() < 0.5
        };
        assert!(
            on_cut,
            "{name}: the segment before the cut at {at} ends elsewhere: {a} {b}"
        );
    }
    let mut turns: Vec<usize> = owners.iter().flatten().copied().collect();
    turns.dedup();
    let speakers: Vec<&str> = segments
        .iter()
        .filter_map(|s| s["speaker"].as_str())
        .collect();
    say(&format!(
        "{name}: {} segments, {changes} speaker changes, {} cuts, turns {turns:?}, speakers {speakers:?}",
        segments.len(),
        cuts.len()
    ));
    assert_eq!(changes, turns.len().saturating_sub(1), "{name}");
}

#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the local speech checkpoints and a Metal device"]
async fn chunked_transcriptions_end_to_end() {
    let Some((state, diar)) = booted_chunking(&[]) else {
        return;
    };
    for name in ["speech-en", "long-dialogue"] {
        every_format(&state, name).await;
    }
    for name in ["two-speakers", "long-dialogue"] {
        speakers_follow_turns(&state, &diar, name, false).await;
    }
}

/// `name`'s diarized transcript from `state`, as its words.
#[cfg(target_os = "macos")]
async fn diarized_words(state: &Arc<AppState>, name: &str) -> Vec<String> {
    let (status, body) = post(
        state,
        form(
            &clip(name),
            &[("model", "qwen3"), ("response_format", "diarized_json")],
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let json: Value = serde_json::from_str(&body).expect("json");
    json["text"]
        .as_str()
        .expect("text")
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

/// The offline path, every non-streaming request sent through it: every
/// format answers, diarized segments change speaker only on the plan's cuts,
/// and a diarized transcript is the chunked stream's word for word where
/// the plan skipped nothing.
#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the local speech checkpoints and a Metal device"]
async fn offline_transcriptions_end_to_end() {
    let Some((state, diar)) = booted_chunking(&["--asr-offline-min-secs", "0"]) else {
        return;
    };
    for name in ["speech-en", "long-dialogue"] {
        every_format(&state, name).await;
    }
    let names = ["two-speakers", "long-dialogue"];
    let mut offline = Vec::new();
    for name in names {
        speakers_follow_turns(&state, &diar, name, true).await;
        offline.push(diarized_words(&state, name).await);
    }
    drop(state);
    let Some((streamed, _)) = booted_chunking(&[]) else {
        return;
    };
    for (name, offline) in names.into_iter().zip(offline) {
        let (_, owners) = turn_plan(&diar, name, true);
        let streamed = diarized_words(&streamed, name).await;
        let (activity, mel) = activity_and_mel(&diar, name);
        let skipped = shiso_audio::chunks::plan_offline(
            &mel,
            Some(&activity),
            0.01,
            &shiso_audio::chunks::OfflineRule::for_10ms_mel(true),
        )
        .expect("plans")
        .iter()
        .filter(|s| !s.transcribe)
        .count();
        say(&format!(
            "{name}: offline {} words, streamed {} words, {skipped} skipped spans of {}",
            offline.len(),
            streamed.len(),
            owners.len()
        ));
        if skipped == 0 {
            assert_eq!(offline, streamed, "{name}");
        }
    }
}
