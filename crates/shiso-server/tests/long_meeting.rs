//! The 57-minute private meeting recording, run through every speech
//! recognizer end to end, Metal only.
//!
//! **Privacy.** The recording and its export never enter this repository.
//! The test reads their paths from `SHISO_TEST_LONG_AUDIO` (the `m4a`) and
//! `SHISO_TEST_LONG_AUDIO_REF` (another system's diarized export: segments
//! with speaker names, and a `full_text`), and skips on stderr when either is
//! unset. Nothing from either file - no text, no speaker name, no excerpt -
//! is ever printed, asserted into a message, or written anywhere; only
//! aggregate counts, times and percentages are.
//!
//! The checkpoints are read from the same roots as `transcription_e2e.rs`:
//! `SHISO_TEST_MODEL_QWEN3ASR`, `SHISO_TEST_MODEL_R2T2`,
//! `SHISO_TEST_MODEL_ASR` (Nemotron) and `SHISO_TEST_MODEL_DIAR`; a
//! recognizer whose root is unset or lacks its file is skipped on stderr.
//! `SHISO_TEST_FFMPEG` names the `ffmpeg` that decodes the `m4a`, `ffmpeg` on
//! `PATH` by default.
//!
//! The export is another system's output, not ground truth: it catches
//! regressions and compares recognizers, so only a floor is asserted (word
//! error rate under 50%, speaker agreement over 50%) for the Qwen3 family.
//! Nemotron's numbers are printed for comparison only.

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::Value;
use shiso_server::cli::{Cli, Command};
use shiso_server::router::build_router;
use shiso_server::state::AppState;
use tower::ServiceExt as _;

#[expect(
    clippy::print_stderr,
    reason = "a skip reason and aggregates go to stderr"
)]
fn say(line: &str) {
    eprintln!("{line}");
}

const BOUNDARY: &str = "shiso-long-meeting-boundary";

/// The MLX diarizer every recognizer in this test shares, under
/// `SHISO_TEST_MODEL_DIAR`; `None` having said why on stderr.
fn diarizer() -> Option<PathBuf> {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_DIAR") else {
        say("SKIP: SHISO_TEST_MODEL_DIAR is unset");
        return None;
    };
    let path = Path::new(&root).join("mlx");
    if !path.exists() {
        say(&format!("SKIP: the diarizer {} is absent", path.display()));
        return None;
    }
    Some(path)
}

/// The decoder for the meeting's `m4a` container: `SHISO_TEST_FFMPEG`, or
/// `ffmpeg` on `PATH`.
fn ffmpeg() -> String {
    std::env::var("SHISO_TEST_FFMPEG").unwrap_or_else(|_| "ffmpeg".to_owned())
}

/// One recognizer this test runs the meeting through.
struct Recognizer {
    /// The name it answers to in the transcription's `model` field.
    name: &'static str,
    /// The variable naming its model root.
    root: &'static str,
    /// Its checkpoint under that root: a GGUF file (mmproj auto-picked) or an
    /// export directory.
    model: &'static str,
    /// Whether its word error rate and speaker agreement are asserted, not
    /// only printed. Nemotron is a comparison point, not a target.
    assert_floor: bool,
}

const RECOGNIZERS: &[Recognizer] = &[
    Recognizer {
        name: "qwen3-asr-q8",
        root: "SHISO_TEST_MODEL_QWEN3ASR",
        model: "gguf/Qwen3-ASR-1.7B-Q8_0.gguf",
        assert_floor: true,
    },
    Recognizer {
        name: "qwen3-asr-mlx",
        root: "SHISO_TEST_MODEL_QWEN3ASR",
        model: "mlx",
        assert_floor: true,
    },
    Recognizer {
        name: "r2t2-q8",
        root: "SHISO_TEST_MODEL_R2T2",
        model: "gguf/Confucius4-R2T2-Q8_0.gguf",
        assert_floor: true,
    },
    Recognizer {
        name: "nemotron-q8",
        root: "SHISO_TEST_MODEL_ASR",
        model: "nvidia/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
        assert_floor: false,
    },
];

/// The paths the private files live at, or `None` having said why on stderr.
fn long_paths() -> Option<(PathBuf, PathBuf)> {
    let (Some(audio), Some(reference)) = (
        std::env::var_os("SHISO_TEST_LONG_AUDIO").map(PathBuf::from),
        std::env::var_os("SHISO_TEST_LONG_AUDIO_REF").map(PathBuf::from),
    ) else {
        say("SKIP: SHISO_TEST_LONG_AUDIO and SHISO_TEST_LONG_AUDIO_REF are needed");
        return None;
    };
    for path in [&audio, &reference] {
        if !path.exists() {
            say(&format!("SKIP: {} is absent", path.display()));
            return None;
        }
    }
    Some((audio, reference))
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
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; \
             filename=\"meeting.m4a\"\r\nContent-Type: audio/mp4\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(audio);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    body
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
    // `Result::expect` would Debug-format a `FromUtf8Error`, which carries the
    // invalid bytes - here, the model's transcript of the private meeting.
    // `{error}` uses `Display`, which carries no bytes.
    let text = String::from_utf8(bytes.to_vec()).unwrap_or_else(|error| panic!("text: {error}"));
    (status, text)
}

/// A recognizer and the shared MLX diarizer, booted as the binary boots them,
/// with `--audio-allow-ffmpeg` and a `--audio-max-seconds` wide enough for the
/// meeting. `None` having said on stderr why the checkpoint was skipped.
fn boot(recognizer: &Recognizer, diarizer: &Path) -> Option<Arc<AppState>> {
    let Some(root) = std::env::var_os(recognizer.root) else {
        say(&format!(
            "SKIP {}: {} is unset",
            recognizer.name, recognizer.root
        ));
        return None;
    };
    let model = Path::new(&root).join(recognizer.model);
    if !model.exists() {
        say(&format!(
            "SKIP {}: {} is absent",
            recognizer.name,
            model.display()
        ));
        return None;
    }
    let (model, diarizer, ffmpeg) = (
        model.display().to_string(),
        diarizer.display().to_string(),
        ffmpeg(),
    );
    let line = [
        "shiso",
        "serve",
        "--asr-model",
        &model,
        "--asr-id",
        recognizer.name,
        "--diarization-model",
        &diarizer,
        "--audio-allow-ffmpeg",
        "--audio-max-seconds",
        "3600",
        "--video-ffmpeg-path",
        &ffmpeg,
        // The default 20 s watchdog is sized for short video clips, not a
        // 57-minute container; ffmpeg's own decode is well under this, but
        // the margin costs nothing.
        "--video-decode-timeout-s",
        "300",
    ];
    let cli = <Cli as clap::Parser>::try_parse_from(line).expect("the command line parses");
    let Command::Serve(args) = cli.command;
    args.validate().expect("servable");
    let audio = shiso_server::boot_audio(&args)
        .unwrap_or_else(|error| panic!("{}: {error}", recognizer.name))
        .expect("an ASR was named");
    Some(Arc::new(
        AppState::without_language_model().with_audio(audio),
    ))
}

/// Lowercases `text` with Turkish casing (`I`/`ı`, `İ`/`i` mapped before the
/// generic lowercasing, which would otherwise fold both onto ASCII `i`), then
/// strips everything but word characters and whitespace and splits on
/// whitespace. Mirrors the scoring approach validated against this export
/// during development.
fn normalize_words(text: &str) -> Vec<String> {
    let turkish: String = text
        .chars()
        .map(|c| match c {
            'I' => 'ı',
            'İ' => 'i',
            other => other,
        })
        .collect();
    let lowered = turkish.to_lowercase();
    let stripped: String = lowered
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    stripped.split_whitespace().map(str::to_owned).collect()
}

/// The word-level edit distance between `reference` and `hypothesis`: the
/// classic single-row dynamic program, insertion, deletion and substitution
/// all costing one word.
fn edit_distance(reference: &[String], hypothesis: &[String]) -> usize {
    let mut d: Vec<usize> = (0..=hypothesis.len()).collect();
    for a in reference {
        let mut diag = d[0];
        d[0] += 1;
        for (col, b) in (1..=hypothesis.len()).zip(hypothesis) {
            let up = d[col];
            let left = d[col - 1];
            let cost = usize::from(a != b);
            d[col] = (up + 1).min(left + 1).min(diag + cost);
            diag = up;
        }
    }
    d[hypothesis.len()]
}

/// A speaker's labelled span: its label and `[start, end)` in seconds.
type Span = (String, f64, f64);

/// The speaker whose span covers `t`: the one with the latest start at or
/// before `t` whose end is still ahead of it.
fn speaker_at(spans: &[Span], t: f64) -> Option<&str> {
    spans
        .iter()
        .filter(|s| s.1 <= t)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .filter(|s| t < s.2)
        .map(|s| s.0.as_str())
}

/// Every ordered selection of `k` distinct items from `items`, order
/// mattering: the partial permutations tried as candidate one-to-one
/// assignments.
fn k_arrangements(items: &[String], k: usize) -> Vec<Vec<String>> {
    fn rec(
        items: &[String],
        k: usize,
        used: &mut [bool],
        current: &mut Vec<String>,
        out: &mut Vec<Vec<String>>,
    ) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for (i, item) in items.iter().enumerate() {
            if used[i] {
                continue;
            }
            used[i] = true;
            current.push(item.clone());
            rec(items, k, used, current, out);
            current.pop();
            used[i] = false;
        }
    }
    let mut used = vec![false; items.len()];
    let mut out = Vec::new();
    rec(items, k, &mut used, &mut Vec::new(), &mut out);
    out
}

/// Every unordered subset of `k` items from `items`.
fn k_combinations(items: &[String], k: usize) -> Vec<Vec<String>> {
    fn rec(
        items: &[String],
        start: usize,
        k: usize,
        current: &mut Vec<String>,
        out: &mut Vec<Vec<String>>,
    ) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for i in start..items.len() {
            current.push(items[i].clone());
            rec(items, i + 1, k, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    rec(items, 0, k, &mut Vec::new(), &mut out);
    out
}

/// Diarization agreement: half-second samples over the export's time range,
/// paired (export speaker, our speaker) wherever both are present and the
/// export speaker is not `UNKNOWN`, scored under the best one-to-one
/// assignment of export names to our labels. Brute force over the small
/// permutation space; it handles either side having more speakers than the
/// other. Returns `(best matched pairs, total pairs)`.
fn speaker_agreement(reference: &[Span], ours: &[Span]) -> (usize, usize) {
    let last_end = reference.iter().map(|s| s.2).fold(0.0_f64, f64::max);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a recording's length in half-seconds is a small non-negative whole number"
    )]
    let samples = (last_end * 2.0) as usize;
    let mut pairs = Vec::new();
    for x in 0..samples {
        let t = f64::from(u32::try_from(x).unwrap_or(u32::MAX)) * 0.5;
        let (Some(r), Some(h)) = (speaker_at(reference, t), speaker_at(ours, t)) else {
            continue;
        };
        if r == "UNKNOWN" {
            continue;
        }
        pairs.push((r.to_owned(), h.to_owned()));
    }
    if pairs.is_empty() {
        return (0, 0);
    }
    let mut counts: HashMap<(String, String), usize> = HashMap::new();
    for pair in &pairs {
        *counts.entry(pair.clone()).or_insert(0) += 1;
    }
    let mut ref_names: Vec<String> = pairs.iter().map(|p| p.0.clone()).collect();
    ref_names.sort_unstable();
    ref_names.dedup();
    let mut hyp_names: Vec<String> = pairs.iter().map(|p| p.1.clone()).collect();
    hyp_names.sort_unstable();
    hyp_names.dedup();
    let k = ref_names.len().min(hyp_names.len());
    let mut best = 0;
    for combo in k_combinations(&ref_names, k) {
        for arrangement in k_arrangements(&hyp_names, k) {
            let score: usize = combo
                .iter()
                .zip(&arrangement)
                .map(|(r, h)| counts.get(&(r.clone(), h.clone())).copied().unwrap_or(0))
                .sum();
            best = best.max(score);
        }
    }
    (best, pairs.len())
}

/// The export's normalised words and speaker spans, read once and never
/// printed: only aggregate counts derived from it are.
struct Reference {
    /// `full_text`, normalised.
    words: Vec<String>,
    /// The segments' `(speaker, start, end)`, `UNKNOWN` included (excluded
    /// only where agreement is scored).
    spans: Vec<Span>,
}

fn load_reference(path: &Path) -> Reference {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let value: Value = serde_json::from_str(&raw).expect("the export is valid json");
    let full_text = value["full_text"].as_str().expect("full_text").to_owned();
    let words = normalize_words(&full_text);
    let segments = value["segments"]["segments"]
        .as_array()
        .expect("segments.segments")
        .iter()
        .map(|s| {
            (
                s["speaker"].as_str().expect("speaker").to_owned(),
                s["start"].as_f64().expect("start"),
                s["end"].as_f64().expect("end"),
            )
        })
        .collect();
    Reference {
        words,
        spans: segments,
    }
}

/// Our own diarized segments, as `(speaker, start, end)`.
fn our_spans(json: &Value) -> Vec<Span> {
    json["segments"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|s| {
            Some((
                s["speaker"].as_str()?.to_owned(),
                s["start"].as_f64()?,
                s["end"].as_f64()?,
            ))
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the private long meeting recording and a Metal device"]
async fn long_meeting_recognizers() {
    let Some((audio_path, reference_path)) = long_paths() else {
        return;
    };
    if !shiso_metal::metal_available() {
        say("SKIP: no Metal device");
        return;
    }
    let Some(diarizer) = diarizer() else {
        return;
    };

    let audio_bytes =
        std::fs::read(&audio_path).unwrap_or_else(|error| panic!("reading the meeting: {error}"));
    let reference = load_reference(&reference_path);
    say(&format!(
        "reference: {} words, {} segments",
        reference.words.len(),
        reference.spans.len()
    ));

    for recognizer in RECOGNIZERS {
        let Some(state) = boot(recognizer, &diarizer) else {
            continue;
        };
        let started = Instant::now();
        let (status, body) = post(
            &state,
            form(
                &audio_bytes,
                &[
                    ("model", recognizer.name),
                    ("response_format", "diarized_json"),
                    ("language", "tr"),
                ],
            ),
        )
        .await;
        let elapsed = started.elapsed().as_secs_f64();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}: request failed",
            recognizer.name
        );
        let json: Value = serde_json::from_str(&body).expect("json");
        let duration = json["duration"].as_f64().unwrap_or(0.0);
        let hyp_words = normalize_words(json["text"].as_str().expect("text"));
        let edits = edit_distance(&reference.words, &hyp_words);
        #[expect(
            clippy::cast_precision_loss,
            reason = "a word count fits a meeting's transcript; f64 carries it exactly"
        )]
        let wer = edits as f64 / reference.words.len().max(1) as f64;
        let spans = our_spans(&json);
        let mut speakers: Vec<&str> = spans.iter().map(|s| s.0.as_str()).collect();
        speakers.sort_unstable();
        speakers.dedup();
        let (matched, pairs) = speaker_agreement(&reference.spans, &spans);
        #[expect(
            clippy::cast_precision_loss,
            reason = "a half-second sample count fits a meeting's length; f64 carries it exactly"
        )]
        let agreement = if pairs == 0 {
            0.0
        } else {
            matched as f64 / pairs as f64
        };
        say(&format!(
            "{}: {elapsed:.1} s wall ({:.3}x realtime over {duration:.1} s), \
             WER {:.1}% ({edits} of {} words), {} speakers found, \
             agreement {:.1}% over {pairs} half-second samples",
            recognizer.name,
            elapsed / duration.max(1.0),
            wer * 100.0,
            reference.words.len(),
            speakers.len(),
            agreement * 100.0,
        ));
        if recognizer.assert_floor {
            assert!(
                wer < 0.5,
                "{}: WER {:.1}% is not under the 50% floor",
                recognizer.name,
                wer * 100.0
            );
            assert!(
                agreement > 0.5,
                "{}: speaker agreement {:.1}% is not over the 50% floor",
                recognizer.name,
                agreement * 100.0
            );
        }
    }
}
