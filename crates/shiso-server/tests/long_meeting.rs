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
//!
//! The meeting is far past `--asr-offline-min-secs`, so a non-streaming
//! request to the Qwen3 family takes the batched offline path. Each of those
//! recognizers runs twice, with `--asr-skip-silence` on (the default) and
//! off, so what skipping long silences costs in words and saves in time is
//! measured on one recording; Nemotron has no offline path and runs once.
//!
//! `SHISO_TEST_KEYWORDS` is optional, widening the check: when it names a
//! file, one `keywords[]` field a line is added to every recognizer whose
//! family accepts a context, and its aggregate line says `keywords=on`.
//! Unset, the run is the plain baseline. A recognizer with no context
//! encoder (Nemotron) never receives them, since it refuses one outright.
//!
//! `SHISO_TEST_DIAG` is optional and separate from the run above: when set,
//! the ordinary recognizer loop is skipped entirely and this test instead
//! diagnoses the keyword-driven WER rise the wider check found, on
//! qwen3-asr-q8 alone, offline with silence kept (the worse of the two
//! silence settings), once without keywords and once with. Unset, this
//! test's output is exactly what it was before this diagnostic existed.

#![cfg(target_os = "macos")]

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::wer::{edit_distance, edit_ops, normalize_words};
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

/// One `keywords[]` field a non-empty line of `SHISO_TEST_KEYWORDS`; empty,
/// so a request built from it sends none, when the variable is unset.
fn keyword_fields() -> Vec<(String, String)> {
    let Some(path) = std::env::var_os("SHISO_TEST_KEYWORDS").map(PathBuf::from) else {
        return Vec::new();
    };
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| ("keywords[]".to_owned(), line.to_owned()))
        .collect()
}

/// Non-overlapping occurrences of `needle` (normalised words) in `hay`. The
/// same rule `keyword_bias.rs` counts with, duplicated rather than shared:
/// this file's diagnostic needs it standalone, the way its other helpers are
/// already duplicated across the two files.
fn occurrences(hay: &[String], needle: &[String]) -> usize {
    if needle.is_empty() || hay.len() < needle.len() {
        return 0;
    }
    let (mut i, mut n) = (0, 0);
    while i + needle.len() <= hay.len() {
        if hay[i..i + needle.len()] == *needle {
            n += 1;
            i += needle.len();
        } else {
            i += 1;
        }
    }
    n
}

/// Words of `hay` covered by non-overlapping occurrences of `needle`; a
/// keyword matched twice covers twice its own word count. Keywords that
/// overlap in the text can double-count a word, which is fine for the
/// diagnostic's "roughly how much of this segment is keywords" reading.
fn covered_words(hay: &[String], needle: &[String]) -> usize {
    if needle.is_empty() || hay.len() < needle.len() {
        return 0;
    }
    let (mut i, mut covered) = (0, 0);
    while i + needle.len() <= hay.len() {
        if hay[i..i + needle.len()] == *needle {
            covered += needle.len();
            i += needle.len();
        } else {
            i += 1;
        }
    }
    covered
}

/// Keyword recall and insertions of `ours` against `reference`: the same
/// per-keyword `min`/`saturating_sub` rule `keyword_bias.rs` scores with.
fn keyword_score(
    reference: &[String],
    ours: &[String],
    keywords: &[Vec<String>],
) -> (usize, usize, usize) {
    let (mut hits, mut total, mut inserted) = (0, 0, 0);
    for k in keywords {
        let (r, o) = (occurrences(reference, k), occurrences(ours, k));
        hits += o.min(r);
        total += r;
        inserted += o.saturating_sub(r);
    }
    (hits, total, inserted)
}

/// Whether `words` (one segment, normalised) reads as a keyword echo: three
/// or more distinct keywords matched anywhere in it, or at least half its
/// words covered by keyword matches.
fn is_echo_segment(words: &[String], keywords: &[Vec<String>]) -> bool {
    if words.is_empty() {
        return false;
    }
    let distinct = keywords
        .iter()
        .filter(|k| occurrences(words, k) >= 1)
        .count();
    if distinct >= 3 {
        return true;
    }
    let covered: usize = keywords.iter().map(|k| covered_words(words, k)).sum();
    #[expect(
        clippy::cast_precision_loss,
        reason = "a segment's word count fits a f64 exactly"
    )]
    let coverage = covered as f64 / words.len() as f64;
    coverage >= 0.5
}

/// Offline spans decoded again for echoing the keywords, and the spans
/// transcribed, summed from the offline job's closing debug event while the
/// diagnostic runs ([`RedecodeCounter`]).
static REDECODED: AtomicU64 = AtomicU64::new(0);
static JOB_SPANS: AtomicU64 = AtomicU64::new(0);

/// Sums the `redecoded` and `spans` fields of the offline job's debug
/// events into [`REDECODED`] and [`JOB_SPANS`]: the only place the server
/// says how many spans it decoded again, since the response does not.
struct RedecodeCounter;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for RedecodeCounter {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        /// The two counts one event carries.
        #[derive(Default)]
        struct Counts {
            redecoded: Option<u64>,
            spans: Option<u64>,
        }
        impl tracing::field::Visit for Counts {
            fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                match field.name() {
                    "redecoded" => self.redecoded = Some(value),
                    "spans" => self.spans = Some(value),
                    _ => {}
                }
            }
            fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
        }
        let mut counts = Counts::default();
        event.record(&mut counts);
        if let (Some(redecoded), Some(spans)) = (counts.redecoded, counts.spans) {
            REDECODED.fetch_add(redecoded, Ordering::Relaxed);
            JOB_SPANS.fetch_add(spans, Ordering::Relaxed);
        }
    }
}

/// Installs [`RedecodeCounter`] for the offline job's debug events alone;
/// whether it could (another subscriber may already be the global one).
fn count_redecodes() -> bool {
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::{Layer as _, filter::Targets};
    let filter =
        Targets::new().with_target("shiso_models::qwen3_asr::offline", tracing::Level::DEBUG);
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(RedecodeCounter.with_filter(filter)),
    )
    .is_ok()
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
    /// Whether it transcribes a long clip offline, so it runs with silence
    /// skipped and kept.
    offline: bool,
    /// Whether its family carries a context encoder, so `keywords[]` may be
    /// sent to it; Nemotron has none and refuses a non-empty context
    /// outright.
    supports_context: bool,
}

const RECOGNIZERS: &[Recognizer] = &[
    Recognizer {
        name: "qwen3-asr-q8",
        root: "SHISO_TEST_MODEL_QWEN3ASR",
        model: "gguf/Qwen3-ASR-1.7B-Q8_0.gguf",
        assert_floor: true,
        offline: true,
        supports_context: true,
    },
    Recognizer {
        name: "qwen3-asr-mlx",
        root: "SHISO_TEST_MODEL_QWEN3ASR",
        model: "mlx",
        assert_floor: true,
        offline: true,
        supports_context: true,
    },
    Recognizer {
        name: "r2t2-q8",
        root: "SHISO_TEST_MODEL_R2T2",
        model: "gguf/Confucius4-R2T2-Q8_0.gguf",
        assert_floor: true,
        offline: true,
        supports_context: true,
    },
    Recognizer {
        name: "nemotron-q8",
        root: "SHISO_TEST_MODEL_ASR",
        model: "nvidia/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
        assert_floor: false,
        offline: false,
        supports_context: false,
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
/// with `--audio-allow-ffmpeg`, a `--audio-max-seconds` wide enough for the
/// meeting, and `extra` flags. `None` having said on stderr why the
/// checkpoint was skipped.
fn boot(recognizer: &Recognizer, diarizer: &Path, extra: &[&str]) -> Option<Arc<AppState>> {
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
    let mut line = vec![
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
    line.extend_from_slice(extra);
    let cli = <Cli as clap::Parser>::try_parse_from(line).expect("the command line parses");
    let Command::Serve(args) = cli.command else {
        panic!("expected Command::Serve");
    };
    args.validate().expect("servable");
    let audio = shiso_server::boot_audio(&args)
        .unwrap_or_else(|error| panic!("{}: {error}", recognizer.name))
        .expect("an ASR was named");
    Some(Arc::new(
        AppState::without_language_model().with_audio(audio),
    ))
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

/// Our own diarized segments' raw `text`, in order.
fn our_segment_texts(json: &Value) -> Vec<String> {
    json["segments"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|s| s["text"].as_str().map(str::to_owned))
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

    if std::env::var_os("SHISO_TEST_DIAG").is_some() {
        diagnose_keyword_wer(&diarizer, &audio_bytes, &reference).await;
        return;
    }

    let keywords = keyword_fields();
    for recognizer in RECOGNIZERS {
        let runs: &[(&str, &[&str])] = if recognizer.offline {
            &[
                ("offline, silence skipped", &["--asr-skip-silence=true"]),
                ("offline, silence kept", &["--asr-skip-silence=false"]),
            ]
        } else {
            &[("streamed", &[])]
        };
        let context: &[(String, String)] = if recognizer.supports_context {
            &keywords
        } else {
            &[]
        };
        for &(run, extra) in runs {
            let Some(state) = boot(recognizer, &diarizer, extra) else {
                break;
            };
            let (wer, agreement) = measure(
                recognizer.name,
                run,
                &state,
                &audio_bytes,
                &reference,
                context,
            )
            .await;
            if recognizer.assert_floor {
                assert!(
                    wer < 0.5,
                    "{} ({run}): WER {:.1}% is not under the 50% floor",
                    recognizer.name,
                    wer * 100.0
                );
                assert!(
                    agreement > 0.5,
                    "{} ({run}): speaker agreement {:.1}% is not over the 50% floor",
                    recognizer.name,
                    agreement * 100.0
                );
            }
        }
    }
}

/// One diarized transcription of the meeting through `state`, `context`
/// added as `keywords[]` fields when non-empty, its aggregates said on
/// stderr; returns its word error rate and speaker agreement.
async fn measure(
    name: &str,
    run: &str,
    state: &Arc<AppState>,
    audio_bytes: &[u8],
    reference: &Reference,
    context: &[(String, String)],
) -> (f64, f64) {
    let started = Instant::now();
    let mut fields: Vec<(&str, &str)> = vec![
        ("model", name),
        ("response_format", "diarized_json"),
        ("language", "tr"),
    ];
    fields.extend(context.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    let (status, body) = post(state, form(audio_bytes, &fields)).await;
    let elapsed = started.elapsed().as_secs_f64();
    assert_eq!(status, StatusCode::OK, "{name} ({run}): request failed");
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
    let keyword_tag = if context.is_empty() {
        ""
    } else {
        " keywords=on"
    };
    say(&format!(
        "{name} ({run}): {elapsed:.1} s wall ({:.3}x realtime over {duration:.1} s), \
         WER {:.1}% ({edits} of {} words; {} words transcribed), {} speakers found, \
         agreement {:.1}% over {pairs} half-second samples{keyword_tag}",
        elapsed / duration.max(1.0),
        wer * 100.0,
        reference.words.len(),
        hyp_words.len(),
        speakers.len(),
        agreement * 100.0,
    ));
    (wer, agreement)
}

/// Diagnoses the WER rise `SHISO_TEST_KEYWORDS` causes on the full meeting:
/// qwen3-asr-q8, offline with silence kept (the worse of the two silence
/// settings measured by the ordinary run), booted once and sent two
/// requests - one without keywords, one with. Each transcript's word errors
/// are split into substitutions, deletions and insertions; its keyword
/// recall and insertions are scored the way `keyword_bias.rs` scores them,
/// against the same export `measure` compares to; and its segments are
/// checked for "keyword echo" (three or more distinct keywords matched, or
/// at least half the segment's words covered by one) and for an exact
/// repeat of the previous segment's text. `rule_echo_segments` counts the
/// segments the recognizer's own rule ([`shiso_audio::is_echo`]) would call
/// an echo, and `redecoded_spans` how many of the offline job's spans it
/// decoded again without the context, out of those it transcribed, read from
/// its debug event. One `diag` line a run, aggregates only. SKIPs on
/// stderr, needing no state, when `SHISO_TEST_KEYWORDS` is unset - the two
/// runs would otherwise be identical - or when qwen3-asr-q8 itself is
/// unavailable (`boot` has already said why).
async fn diagnose_keyword_wer(diarizer: &Path, audio_bytes: &[u8], reference: &Reference) {
    let keyword_lines = keyword_fields();
    if keyword_lines.is_empty() {
        say("SKIP diag: SHISO_TEST_KEYWORDS is needed for the with/without comparison");
        return;
    }
    let keyword_needles: Vec<Vec<String>> = keyword_lines
        .iter()
        .map(|(_, value)| normalize_words(value))
        .collect();
    // The production rule's split of each keyword, for `rule_echo_segments`.
    let rule_needles: Vec<Vec<String>> = keyword_lines
        .iter()
        .map(|(_, value)| shiso_audio::keyword_words(value))
        .collect();
    let counting = count_redecodes();
    // By name rather than `RECOGNIZERS[0]`, so a reordering of that list
    // cannot silently point this diagnostic at the wrong recognizer in a
    // release build, where a `debug_assert!` would have compiled out.
    let recognizer = RECOGNIZERS
        .iter()
        .find(|r| r.name == "qwen3-asr-q8")
        .expect("qwen3-asr-q8 is in RECOGNIZERS");
    let Some(state) = boot(recognizer, diarizer, &["--asr-skip-silence=false"]) else {
        return;
    };

    for (run, context) in [
        ("baseline", [].as_slice()),
        ("keywords", keyword_lines.as_slice()),
    ] {
        let mut fields: Vec<(&str, &str)> = vec![
            ("model", recognizer.name),
            ("response_format", "diarized_json"),
            ("language", "tr"),
        ];
        fields.extend(context.iter().map(|(k, v)| (k.as_str(), v.as_str())));
        REDECODED.store(0, Ordering::Relaxed);
        JOB_SPANS.store(0, Ordering::Relaxed);
        let (status, body) = post(&state, form(audio_bytes, &fields)).await;
        assert_eq!(status, StatusCode::OK, "diag {run}: request failed");
        let redecoded = if counting {
            format!(
                "{}/{}",
                REDECODED.load(Ordering::Relaxed),
                JOB_SPANS.load(Ordering::Relaxed)
            )
        } else {
            "unobserved".to_owned()
        };
        let json: Value = serde_json::from_str(&body).expect("json");
        let hyp_words = normalize_words(json["text"].as_str().expect("text"));
        let (subs, dels, ins) = edit_ops(&reference.words, &hyp_words);
        #[expect(
            clippy::cast_precision_loss,
            reason = "a word count fits a meeting's transcript; f64 carries it exactly"
        )]
        let wer = (subs + dels + ins) as f64 / reference.words.len().max(1) as f64;
        let (keyword_hits, keyword_total, keyword_insertions) =
            keyword_score(&reference.words, &hyp_words, &keyword_needles);

        let (mut echo_segments, mut echo_words) = (0usize, 0usize);
        let (mut repeat_segments, mut repeat_words) = (0usize, 0usize);
        let mut rule_echo_segments = 0usize;
        let mut previous: Option<String> = None;
        for text in our_segment_texts(&json) {
            let trimmed = text.trim();
            let words = normalize_words(trimmed);
            if is_echo_segment(&words, &keyword_needles) {
                echo_segments += 1;
                echo_words += words.len();
            }
            if shiso_audio::is_echo(trimmed, &rule_needles) {
                rule_echo_segments += 1;
            }
            if previous.as_deref() == Some(trimmed) {
                repeat_segments += 1;
                repeat_words += words.len();
            }
            previous = Some(trimmed.to_owned());
        }

        say(&format!(
            "diag run={run} wer={:.1}% subs={subs} dels={dels} ins={ins} \
             keyword_recall={keyword_hits}/{keyword_total} keyword_insertions={keyword_insertions} \
             words={} echo_segments={echo_segments} echo_words={echo_words} \
             repeat_segments={repeat_segments} repeat_words={repeat_words} \
             rule_echo_segments={rule_echo_segments} redecoded_spans={redecoded}",
            wer * 100.0,
            hyp_words.len(),
        ));
    }
}
