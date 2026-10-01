//! How the shape of `keywords[]` and `prompt` affects Qwen3-ASR Q8 on the
//! private short recording, Metal only.
//!
//! **Privacy.** The recording, its hand-corrected reference and the keyword
//! list never enter this repository. The test reads their paths from
//! `SHISO_TEST_SHORT_AUDIO` (the `m4a`), `SHISO_TEST_SHORT_AUDIO_REF` (the
//! hand-corrected transcript) and `SHISO_TEST_KEYWORDS` (one keyword a
//! line), and skips on stderr when any is unset. Nothing from any of the
//! three - no text, no keyword, no excerpt - is ever printed, asserted into
//! a message, or written anywhere; only aggregate word error rate, keyword
//! hit/total counts, insertions and elapsed seconds are.

#![cfg(target_os = "macos")]

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::wer::{edit_distance, normalize_words};
use serde_json::Value;
use shiso_server::cli::{Cli, Command};
use shiso_server::router::build_router;
use shiso_server::state::AppState;
use tower::ServiceExt as _;

#[expect(
    clippy::print_stderr,
    reason = "a skip reason and the per-run aggregate line go to stderr"
)]
fn say(line: &str) {
    eprintln!("{line}");
}

const BOUNDARY: &str = "shiso-keyword-bias-boundary";

/// The three private paths, or `None` having said on stderr which is
/// missing.
fn paths() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let (Some(audio), Some(reference), Some(keywords)) = (
        std::env::var_os("SHISO_TEST_SHORT_AUDIO").map(PathBuf::from),
        std::env::var_os("SHISO_TEST_SHORT_AUDIO_REF").map(PathBuf::from),
        std::env::var_os("SHISO_TEST_KEYWORDS").map(PathBuf::from),
    ) else {
        say(
            "SKIP: SHISO_TEST_SHORT_AUDIO, SHISO_TEST_SHORT_AUDIO_REF and \
             SHISO_TEST_KEYWORDS are needed",
        );
        return None;
    };
    for path in [&audio, &reference, &keywords] {
        if !path.exists() {
            say(&format!("SKIP: {} is absent", path.display()));
            return None;
        }
    }
    Some((audio, reference, keywords))
}

/// Qwen3-ASR Q8 and the shared MLX diarizer, booted with the arguments the
/// production launchd definition passes
/// (`--asr-id qwen3-asr --audio-allow-ffmpeg --asr-batch 4
/// --asr-skip-silence=false --audio-max-seconds 10800`), plus
/// `--asr-offline-min-secs 300` added explicitly so the 327 s sample still
/// takes the offline path the service uses. `None` having said on stderr why
/// a checkpoint was skipped.
fn boot() -> Option<Arc<AppState>> {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN3ASR") else {
        say("SKIP: SHISO_TEST_MODEL_QWEN3ASR is unset");
        return None;
    };
    let model = Path::new(&root).join("gguf/Qwen3-ASR-1.7B-Q8_0.gguf");
    if !model.exists() {
        say(&format!("SKIP: {} is absent", model.display()));
        return None;
    }
    let Some(diar_root) = std::env::var_os("SHISO_TEST_MODEL_DIAR") else {
        say("SKIP: SHISO_TEST_MODEL_DIAR is unset");
        return None;
    };
    let diarizer = Path::new(&diar_root).join("mlx");
    if !diarizer.exists() {
        say(&format!(
            "SKIP: the diarizer {} is absent",
            diarizer.display()
        ));
        return None;
    }
    let (model, diarizer) = (model.display().to_string(), diarizer.display().to_string());
    let cli = <Cli as clap::Parser>::try_parse_from([
        "shiso",
        "serve",
        "--asr-model",
        &model,
        "--asr-id",
        "qwen3-asr",
        "--diarization-model",
        &diarizer,
        "--audio-allow-ffmpeg",
        "--asr-batch",
        "4",
        "--asr-skip-silence=false",
        "--audio-max-seconds",
        "10800",
        "--asr-offline-min-secs",
        "300",
    ])
    .expect("the command line parses");
    let Command::Serve(args) = cli.command else {
        panic!("expected Command::Serve");
    };
    args.validate().expect("servable");
    let audio = shiso_server::boot_audio(&args)
        .unwrap_or_else(|error| panic!("qwen3-asr-q8: {error}"))
        .expect("an ASR was named");
    Some(Arc::new(
        AppState::without_language_model().with_audio(audio),
    ))
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
             filename=\"sample.m4a\"\r\nContent-Type: audio/mp4\r\n\r\n"
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
    // invalid bytes - here, the model's transcript of the private recording.
    // `{error}` uses `Display`, which carries no bytes.
    let text = String::from_utf8(bytes.to_vec()).unwrap_or_else(|error| panic!("text: {error}"));
    (status, text)
}

/// The reference's normalised words: the segments' `text` joined with
/// spaces. The fixture has no `full_text` field, unlike `long_meeting.rs`'s
/// export.
fn load_reference(path: &Path) -> Vec<String> {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("reading the reference: {error}"));
    let value: Value = serde_json::from_str(&raw).expect("the reference is valid json");
    let segments = value["segments"]["segments"]
        .as_array()
        .expect("segments.segments");
    let joined = segments
        .iter()
        .map(|s| s["text"].as_str().expect("a segment's text"))
        .collect::<Vec<_>>()
        .join(" ");
    normalize_words(&joined)
}

/// One keyword a non-empty line of `path`, in file order.
fn load_keywords(path: &Path) -> Vec<String> {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("reading the keyword list: {error}"));
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Non-overlapping occurrences of `needle` (normalised words) in `hay`.
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

/// Keyword recall and insertions of `ours` against `reference`: per
/// keyword, hits are `min(ours, ref)`, insertions `ours - ref` when above.
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

/// The context fields one run sends, `model` aside: `keywords[]` once a
/// keyword and/or a `prompt`, worded the way `run` names.
///
/// These names describe the wire shape each run sends, not the text the
/// model reads: at `HEAD`, `keywords[]` is rendered through
/// `qwen3_asr::context::wording`, which now emits the labelled shape this
/// file measured as `labelled` (that measurement, taken before the fix, is
/// what picked it). So `comma` and `labelled` put the same system text on
/// the wire today, and `prompt+comma` is really prompt-plus-labelled. This
/// file is not re-run to confirm that; the run names are kept as recorded
/// so the numbers in `context.rs`'s doc comment stay traceable to them.
fn context_fields(run: &str, keywords: &[String]) -> Vec<(String, String)> {
    /// The fixed scenario prompt `prompt+comma` pairs with `keywords[]`.
    const SCENARIO_PROMPT: &str =
        "A software team's weekly status meeting, in Turkish with English technical terms.";
    match run {
        "none" => Vec::new(),
        "comma" => keywords
            .iter()
            .map(|k| ("keywords[]".to_owned(), k.clone()))
            .collect(),
        "lines" => vec![("prompt".to_owned(), keywords.join("\n"))],
        "labelled" => vec![(
            "prompt".to_owned(),
            format!("Keywords: {}", keywords.join(", ")),
        )],
        "prompt+comma" => {
            let mut fields = vec![("prompt".to_owned(), SCENARIO_PROMPT.to_owned())];
            fields.extend(
                keywords
                    .iter()
                    .map(|k| ("keywords[]".to_owned(), k.clone())),
            );
            fields
        }
        other => panic!("unknown run {other}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the private short recording and a Metal device"]
async fn keyword_wording_on_the_short_recording() {
    let Some((audio_path, reference_path, keywords_path)) = paths() else {
        return;
    };
    if !shiso_metal::metal_available() {
        say("SKIP: no Metal device");
        return;
    }
    let Some(state) = boot() else {
        return;
    };

    let audio_bytes =
        std::fs::read(&audio_path).unwrap_or_else(|error| panic!("reading the sample: {error}"));
    let reference_words = load_reference(&reference_path);
    let keywords = load_keywords(&keywords_path);
    let keyword_needles: Vec<Vec<String>> = keywords.iter().map(|k| normalize_words(k)).collect();

    let mut none_wer: Option<f64> = None;

    for run in ["none", "comma", "lines", "labelled", "prompt+comma"] {
        let context = context_fields(run, &keywords);
        let mut fields: Vec<(&str, &str)> = vec![("model", "qwen3-asr")];
        fields.extend(context.iter().map(|(k, v)| (k.as_str(), v.as_str())));

        let started = Instant::now();
        let (status, body) = post(&state, form(&audio_bytes, &fields)).await;
        let elapsed = started.elapsed().as_secs_f64();
        assert_eq!(status, StatusCode::OK, "run={run}: request failed");

        let json: Value = serde_json::from_str(&body).expect("json");
        let hyp_words = normalize_words(json["text"].as_str().expect("text"));
        let edits = edit_distance(&reference_words, &hyp_words);
        #[expect(
            clippy::cast_precision_loss,
            reason = "a short sample's word count fits a f64 exactly"
        )]
        let wer = edits as f64 / reference_words.len().max(1) as f64;

        let (hits, total, inserted) = keyword_score(&reference_words, &hyp_words, &keyword_needles);
        #[expect(
            clippy::cast_precision_loss,
            reason = "a keyword count fits a f64 exactly"
        )]
        let recall_pct = if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64 * 100.0
        };

        say(&format!(
            "run={run} wer={:.1}% keyword_recall={hits}/{total} ({recall_pct:.1}%) \
             insertions={inserted} secs={elapsed:.1}",
            wer * 100.0
        ));

        if run == "none" {
            none_wer = Some(wer);
        }
        if run == "comma" {
            let bound = none_wer.expect("none ran first") + 0.01;
            assert!(
                wer <= bound,
                "run=comma: wer {:.1}% exceeds none+1.0pp bound {:.1}%",
                wer * 100.0,
                bound * 100.0
            );
        }
    }
}
