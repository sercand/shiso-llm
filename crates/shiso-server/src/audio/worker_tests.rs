//! The audio worker over a scripted recognizer and diarizer: event order,
//! fairness between jobs, a caller that leaves, a full queue, and a failure.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use shiso_audio::chunks;
use shiso_audio::transcript::SpeakerActivity;
use shiso_audio::{
    AudioError, DiarizationPass, Diarizer, Filterbank, Framing, Language, LogScale, MelFrames,
    MelFrontend, MelScale, MelSpec, Pcm, PieceVocab, Preemphasis, RecognizerStream, Spectrum,
    SpeechRecognizer, StreamOptions, TokenEvent, Window,
};
use shiso_core::TokenId;
use tokio::sync::mpsc::UnboundedReceiver;

use super::{AudioService, AudioServiceDesc, TranscriptEvent, TranscriptionJob};
use crate::error::AppError;

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

/// Which job each push belonged to, in the order the worker made them.
type Log = Arc<Mutex<Vec<String>>>;

/// The options each stream was opened with, in order.
type Opened = Arc<Mutex<Vec<StreamOptions>>>;

/// Emits `▁a` or `▁b` for every 40 mel rows it is fed, alternating, and one
/// more on flush; refuses a stream conditioned on `fail`. With `chunks`, it
/// says it honours cuts, as a chunking recognizer does.
#[derive(Debug)]
struct Scripted {
    spec: MelSpec,
    vocab: Arc<PieceVocab>,
    log: Log,
    chunks: bool,
    opened: Opened,
}

struct ScriptedStream {
    label: String,
    rows: usize,
    emitted: u32,
    log: Log,
}

impl RecognizerStream for ScriptedStream {
    fn push(&mut self, mel: &MelFrames) -> Result<Vec<TokenEvent>, AudioError> {
        self.log
            .lock()
            .expect("unpoisoned")
            .push(self.label.clone());
        assert!(self.label != "panic", "a scripted panic in the recognizer");
        if self.label == "silent" {
            self.rows += mel.valid();
            return Ok(Vec::new());
        }
        if self.label == "fail" {
            return Err(AudioError::BadSpec {
                reason: "scripted failure".into(),
            });
        }
        self.rows += mel.valid();
        let mut out = Vec::new();
        while (self.emitted + 1) as usize * 40 <= self.rows {
            out.push(TokenEvent {
                token: TokenId::new(1 + self.emitted % 2),
                frame: self.emitted * 5,
            });
            self.emitted += 1;
        }
        Ok(out)
    }
    fn finish(&mut self) -> Result<Vec<TokenEvent>, AudioError> {
        self.emitted += 1;
        Ok(vec![TokenEvent {
            token: TokenId::new(1),
            frame: self.emitted * 5,
        }])
    }
    fn frames(&self) -> u32 {
        u32::try_from(self.rows / 8).unwrap_or(u32::MAX)
    }
    fn language(&self) -> Option<&str> {
        // A stream conditioned on `speaks` reads its language out of the
        // model's words, as a model that writes `language X` does.
        (self.label == "speaks").then_some("xx")
    }
}

impl SpeechRecognizer for Scripted {
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
    fn chunks_at_cuts(&self) -> bool {
        self.chunks
    }
    fn open_stream(
        &self,
        options: &StreamOptions,
    ) -> Result<Box<dyn RecognizerStream>, AudioError> {
        self.opened
            .lock()
            .expect("unpoisoned")
            .push(options.clone());
        let label = match &options.language {
            Language::Auto => "auto".to_owned(),
            Language::Tag(tag) => tag.clone(),
        };
        Ok(Box::new(ScriptedStream {
            label,
            rows: 0,
            emitted: 0,
            log: Arc::clone(&self.log),
        }))
    }
}

/// Speaker 0 for the first half of the clip, speaker 1 for the rest, after
/// twenty logged steps of 5 ms each: long enough that recognition running
/// beside it shows in the log.
#[derive(Debug)]
struct Halves {
    spec: MelSpec,
    log: Log,
}

struct HalvesPass {
    frames: usize,
    left: usize,
    log: Log,
}

impl DiarizationPass for HalvesPass {
    // `None`: the pass has more steps to run; it is not a refusal.
    fn step(&mut self) -> Result<Option<SpeakerActivity>, AudioError> {
        self.log.lock().expect("unpoisoned").push("diar".into());
        std::thread::sleep(Duration::from_millis(5));
        if self.left > 0 {
            self.left -= 1;
            return Ok(None);
        }
        let n = self.frames;
        let probs = (0..n)
            .flat_map(|f| if f < n / 2 { [0.9, 0.0] } else { [0.0, 0.9] })
            .collect();
        SpeakerActivity::new(0.01, 2, probs).map(Some)
    }
}

impl Diarizer for Halves {
    fn mel_spec(&self) -> &MelSpec {
        &self.spec
    }
    fn speakers(&self) -> usize {
        2
    }
    fn frame_secs(&self) -> f64 {
        0.01
    }
    fn begin(&self, mel: MelFrames) -> Result<Box<dyn DiarizationPass>, AudioError> {
        Ok(Box::new(HalvesPass {
            frames: mel.valid(),
            left: 20,
            log: Arc::clone(&self.log),
        }))
    }
}

fn service(max_streams: usize, queue: usize, log: &Log) -> AudioService {
    service_with_upload(max_streams, queue, log, super::DEFAULT_MAX_UPLOAD_BYTES)
}

fn service_with_upload(
    max_streams: usize,
    queue: usize,
    log: &Log,
    max_upload_bytes: usize,
) -> AudioService {
    service_with(&ServiceShape {
        max_streams,
        queue,
        log,
        max_upload_bytes,
        chunks: false,
        opened: &Opened::default(),
    })
}

/// What a test service is built with.
struct ServiceShape<'a> {
    max_streams: usize,
    queue: usize,
    log: &'a Log,
    max_upload_bytes: usize,
    chunks: bool,
    opened: &'a Opened,
}

fn service_with(shape: &ServiceShape<'_>) -> AudioService {
    let ServiceShape {
        max_streams,
        queue,
        log,
        max_upload_bytes,
        chunks,
        opened,
    } = *shape;
    let vocab = Arc::new(
        PieceVocab::from_pieces(vec![
            "<blank>".into(),
            "\u{2581}a".into(),
            "\u{2581}b".into(),
        ])
        .expect("pieces"),
    );
    let (service, _join) = AudioService::start(AudioServiceDesc {
        recognizer: Arc::new(Scripted {
            spec: spec(),
            vocab,
            log: Arc::clone(log),
            chunks,
            opened: Arc::clone(opened),
        }),
        diarizer: Some(Arc::new(Halves {
            spec: spec(),
            log: Arc::clone(log),
        })),
        asr_id: "scripted".into(),
        lookahead: 0,
        max_streams,
        queue,
        limits: crate::tokenizer::message_preprocess::DEFAULT_AUDIO_LIMITS,
        ffmpeg: None,
        max_upload_bytes,
    })
    .expect("the thread starts");
    service
}

fn job(seconds: usize, language: &str, diarize: bool) -> TranscriptionJob {
    let samples = (0..seconds * 16_000)
        .map(|i| (f32::from(u16::try_from(i % 97).expect("small")) / 97.0) - 0.5)
        .collect();
    TranscriptionJob {
        pcm: Pcm::new(samples, 16_000).expect("pcm"),
        language: Language::Tag(language.into()),
        diarize,
    }
}

/// Every event of one job, up to and including its last.
async fn drain(mut rx: UnboundedReceiver<TranscriptEvent>) -> Vec<TranscriptEvent> {
    let mut out = Vec::new();
    while let Some(event) = tokio::time::timeout(Duration::from_secs(60), rx.recv())
        .await
        .expect("the worker answers")
    {
        let last = matches!(event, TranscriptEvent::Done(_) | TranscriptEvent::Failed(_));
        out.push(event);
        if last {
            break;
        }
    }
    out
}

#[tokio::test]
async fn a_job_streams_its_text_and_ends_in_the_whole_transcript() {
    let log = Log::default();
    let events = drain(
        service(2, 4, &log)
            .submit(job(2, "a", false))
            .expect("queued"),
    )
    .await;
    let Some(TranscriptEvent::Done(transcript)) = events.last() else {
        panic!("{events:?}");
    };
    let deltas: String = events
        .iter()
        .filter_map(|e| match e {
            TranscriptEvent::Delta(d) => Some(d.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas.trim_end(), transcript.text);
    assert!(transcript.text.starts_with("a b a"), "{}", transcript.text);
    assert!(!transcript.words.is_empty());
    assert!((transcript.duration - 2.0).abs() < 1e-9);
    assert!(!transcript.diarized);
}

#[tokio::test]
async fn a_language_the_stream_reports_reaches_the_transcript() {
    let log = Log::default();
    let svc = service(2, 4, &log);
    for (tag, want) in [("speaks", Some("xx")), ("a", None)] {
        let events = drain(svc.submit(job(1, tag, false)).expect("queued")).await;
        let Some(TranscriptEvent::Done(transcript)) = events.last() else {
            panic!("{events:?}");
        };
        assert_eq!(transcript.language.as_deref(), want, "{tag}");
    }
}

#[tokio::test]
async fn two_long_jobs_advance_together() {
    let log = Log::default();
    let svc = service(2, 4, &log);
    // Long enough to take several turns each at the widest turn a round
    // gives two jobs.
    let a = svc.submit(job(40, "a", false)).expect("queued");
    let b = svc.submit(job(40, "b", false)).expect("queued");
    let (a, b) = tokio::join!(drain(a), drain(b));
    assert!(matches!(a.last(), Some(TranscriptEvent::Done(_))));
    assert!(matches!(b.last(), Some(TranscriptEvent::Done(_))));
    let log = log.lock().expect("unpoisoned").clone();
    let last_a = log.iter().rposition(|l| l == "a").expect("a ran");
    let first_b = log.iter().position(|l| l == "b").expect("b ran");
    assert!(first_b < last_a, "b waited for a to finish: {log:?}");
}

#[tokio::test]
async fn a_caller_that_leaves_frees_its_slot() {
    let log = Log::default();
    let svc = service(1, 4, &log);
    drop(svc.submit(job(8, "gone", false)).expect("queued"));
    let events = drain(svc.submit(job(1, "next", false)).expect("queued")).await;
    assert!(
        matches!(events.last(), Some(TranscriptEvent::Done(_))),
        "{events:?}"
    );
}

#[tokio::test]
async fn a_full_queue_is_refused_as_overloaded() {
    let log = Log::default();
    let svc = service(1, 1, &log);
    let mut refused = None;
    let mut held = Vec::new();
    for _ in 0..3 {
        match svc.submit(job(20, "long", false)) {
            Ok(rx) => held.push(rx),
            Err(error) => refused = Some(error),
        }
    }
    assert!(
        matches!(refused, Some(AppError::Overloaded { .. })),
        "{refused:?}"
    );
}

#[tokio::test]
async fn a_recognizer_failure_ends_the_job_saying_why() {
    let log = Log::default();
    let events = drain(
        service(1, 4, &log)
            .submit(job(1, "fail", false))
            .expect("queued"),
    )
    .await;
    assert!(
        matches!(events.last(), Some(TranscriptEvent::Failed(why)) if why.message.contains("scripted") && !why.caller),
        "{events:?}"
    );
}

#[tokio::test]
async fn a_diarized_job_names_each_segments_speaker() {
    let log = Log::default();
    let events = drain(
        service(1, 4, &log)
            .submit(job(6, "a", true))
            .expect("queued"),
    )
    .await;
    let Some(TranscriptEvent::Done(transcript)) = events.last() else {
        panic!("{events:?}");
    };
    assert!(transcript.diarized);
    let speakers: Vec<Option<usize>> = transcript.segments.iter().map(|s| s.speaker).collect();
    assert_eq!(speakers.first(), Some(&Some(0)), "{speakers:?}");
    assert_eq!(speakers.last(), Some(&Some(1)), "{speakers:?}");
    let streamed = events
        .iter()
        .filter(|e| matches!(e, TranscriptEvent::Segment(_)))
        .count();
    assert_eq!(
        streamed,
        transcript.segments.len() - 1,
        "every segment but the last streams as it settles"
    );
}

#[tokio::test]
async fn an_audio_only_server_lists_its_recognizer() {
    let log = Log::default();
    let state = crate::state::AppState::without_language_model().with_audio(service(1, 1, &log));
    let listing = crate::api::misc_handlers::model_listing(&state);
    let ids: Vec<&str> = listing.data.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, vec!["scripted"]);
}

#[tokio::test]
async fn a_caller_that_leaves_a_silent_job_frees_its_slot_at_once() {
    let log = Log::default();
    let svc = service(1, 4, &log);
    // Silent audio sends no text, so no failed send can tell the worker the
    // caller has gone; it has to look.
    drop(svc.submit(job(60, "silent", false)).expect("queued"));
    let events = drain(svc.submit(job(1, "next", false)).expect("queued")).await;
    assert!(
        matches!(events.last(), Some(TranscriptEvent::Done(_))),
        "{events:?}"
    );
    let silent = log
        .lock()
        .expect("unpoisoned")
        .iter()
        .filter(|l| *l == "silent")
        .count();
    assert!(silent <= 1, "the abandoned job ran {silent} turns");
}

#[tokio::test]
async fn a_job_that_panics_fails_alone() {
    let log = Log::default();
    let svc = service(1, 4, &log);
    let events = drain(svc.submit(job(1, "panic", false)).expect("queued")).await;
    assert!(
        matches!(events.last(), Some(TranscriptEvent::Failed(why)) if !why.caller),
        "{events:?}"
    );
    let events = drain(svc.submit(job(1, "next", false)).expect("queued")).await;
    assert!(
        matches!(events.last(), Some(TranscriptEvent::Done(_))),
        "the worker survived the panic: {events:?}"
    );
}

#[tokio::test]
async fn a_diarized_job_is_recognised_while_it_is_diarized() {
    let log = Log::default();
    let events = drain(
        service(1, 4, &log)
            .submit(job(2, "a", true))
            .expect("queued"),
    )
    .await;
    assert!(matches!(events.last(), Some(TranscriptEvent::Done(_))));
    let log = log.lock().expect("unpoisoned").clone();
    let last_diar = log.iter().rposition(|l| l == "diar").expect("diarized");
    let first_a = log.iter().position(|l| l == "a").expect("a ran");
    assert!(
        first_a < last_diar,
        "recognition waited for the diarizer: {log:?}"
    );
}

#[tokio::test]
async fn a_chunking_recognizer_opens_its_stream_with_the_diarizer_cuts() {
    let log = Log::default();
    let opened = Opened::default();
    let svc = service_with(&ServiceShape {
        max_streams: 1,
        queue: 4,
        log: &log,
        max_upload_bytes: super::DEFAULT_MAX_UPLOAD_BYTES,
        chunks: true,
        opened: &opened,
    });
    let clip = job(6, "a", true);
    // What the worker sees: the recognizer's mel, and the scripted
    // diarizer's activity over the diarizer's own (here identical) mel.
    let mel = MelFrontend::new(spec())
        .expect("spec")
        .compute(clip.pcm.samples());
    let frames = mel.valid();
    let probs = (0..frames)
        .flat_map(|f| {
            if f < frames / 2 {
                [0.9, 0.0]
            } else {
                [0.0, 0.9]
            }
        })
        .collect();
    let activity = SpeakerActivity::new(0.01, 2, probs).expect("geometry");
    let expected: Vec<u32> = chunks::turn_cuts(&activity, &mel, 0.01)
        .into_iter()
        .map(|c| u32::try_from(c).expect("small"))
        .collect();
    assert_eq!(expected.len(), 1, "one turn change: {expected:?}");
    let events = drain(svc.submit(clip).expect("queued")).await;
    let Some(TranscriptEvent::Done(transcript)) = events.last() else {
        panic!("{events:?}");
    };
    assert!(transcript.diarized);
    let opened = opened.lock().expect("unpoisoned").clone();
    assert_eq!(opened.len(), 1, "{opened:?}");
    assert_eq!(opened[0].cuts, Some(expected));
    let log = log.lock().expect("unpoisoned").clone();
    let last_diar = log.iter().rposition(|l| l == "diar").expect("diarized");
    let first_a = log.iter().position(|l| l == "a").expect("a ran");
    assert!(
        last_diar < first_a,
        "rows were pushed before the cuts were known: {log:?}"
    );
}

#[tokio::test]
async fn a_non_chunking_recognizer_is_opened_without_cuts() {
    let log = Log::default();
    let opened = Opened::default();
    let svc = service_with(&ServiceShape {
        max_streams: 1,
        queue: 4,
        log: &log,
        max_upload_bytes: super::DEFAULT_MAX_UPLOAD_BYTES,
        chunks: false,
        opened: &opened,
    });
    let events = drain(svc.submit(job(2, "a", true)).expect("queued")).await;
    assert!(matches!(events.last(), Some(TranscriptEvent::Done(_))));
    let opened = opened.lock().expect("unpoisoned").clone();
    assert_eq!(opened.len(), 1, "{opened:?}");
    assert_eq!(opened[0].cuts, None);
}

#[tokio::test]
async fn a_long_diarization_holds_no_other_job() {
    let log = Log::default();
    let svc = service(2, 4, &log);
    let diarized = svc.submit(job(2, "a", true)).expect("queued");
    let plain = svc.submit(job(40, "b", false)).expect("queued");
    let (diarized, plain) = tokio::join!(drain(diarized), drain(plain));
    assert!(matches!(diarized.last(), Some(TranscriptEvent::Done(_))));
    assert!(matches!(plain.last(), Some(TranscriptEvent::Done(_))));
    let log = log.lock().expect("unpoisoned").clone();
    let last_diar = log.iter().rposition(|l| l == "diar").expect("diarized");
    let first_b = log.iter().position(|l| l == "b").expect("b ran");
    assert!(first_b < last_diar, "b waited for the diarizer: {log:?}");
}

/// A transcription form carrying `file_bytes` bytes that are not audio.
fn upload(file_bytes: usize) -> axum::http::Request<axum::body::Body> {
    let boundary = "shisoboundary";
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nscripted\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.bin\"\r\n\
         Content-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    body.extend(std::iter::repeat_n(7u8, file_bytes));
    body.extend(format!("\r\n--{boundary}--\r\n").into_bytes());
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(axum::body::Body::from(body))
        .expect("builds")
}

async fn status_and_code(
    service: AudioService,
    request: axum::http::Request<axum::body::Body>,
) -> (axum::http::StatusCode, serde_json::Value) {
    use tower::ServiceExt as _;
    let state = Arc::new(crate::state::AppState::without_language_model().with_audio(service));
    let response = crate::router::build_router(state)
        .oneshot(request)
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

#[tokio::test]
async fn an_upload_past_the_json_body_limit_is_read_as_audio() {
    let log = Log::default();
    // 40 MiB: past the 32 MiB every other route takes, inside the audio limit.
    let (status, body) = status_and_code(
        service_with_upload(1, 1, &log, 64 * 1024 * 1024),
        upload(40 * 1024 * 1024),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["code"], "invalid_audio",
        "it was read: {body}"
    );
}

#[tokio::test]
async fn an_upload_past_the_audio_limit_is_refused_as_too_large() {
    let log = Log::default();
    let (status, body) = status_and_code(
        service_with_upload(1, 1, &log, 1024 * 1024),
        upload(2 * 1024 * 1024),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(body["error"]["code"], "request_too_large", "{body}");
}
