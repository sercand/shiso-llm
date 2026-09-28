//! The audio worker over a scripted recognizer and diarizer: event order,
//! fairness between jobs, a caller that leaves, a full queue, a failure, and
//! the offline path of a recognizer that batches.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::http::StatusCode;
use shiso_audio::chunks::{self, OfflineRule, Span};
use shiso_audio::transcript::SpeakerActivity;
use shiso_audio::{
    AudioError, Context, ContextEncoder, ContextIds, DiarizationPass, Diarizer, Filterbank,
    Framing, Language, LogScale, MelFrames, MelFrontend, MelScale, MelSpec, OfflineBatch,
    OfflineJob, Pcm, PieceVocab, Preemphasis, RecognizerStream, SpanResult, Spectrum,
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

/// The plan each offline job was opened with, in order.
type OpenedOffline = Arc<Mutex<Vec<Vec<Span>>>>;

/// Emits `▁a` or `▁b` for every 40 mel rows it is fed, alternating, and one
/// more on flush; refuses a stream conditioned on `fail`. With `chunks`, it
/// says it honours cuts, as a chunking recognizer does. With `offline` above
/// zero it batches that many spans a step ([`ScriptedOffline`]).
#[derive(Debug)]
struct Scripted {
    spec: MelSpec,
    vocab: Arc<PieceVocab>,
    log: Log,
    chunks: bool,
    opened: Opened,
    offline: usize,
    opened_offline: OpenedOffline,
    /// The context encoder this recognizer hands the server, or `None` for a
    /// recognizer that takes no context.
    encoder: Option<Arc<dyn ContextEncoder>>,
}

/// Encodes one id per keyword (`1000 + index`), ignoring the prompt's words;
/// caps at 4 ids.
#[derive(Debug)]
struct ScriptedEncoder;

impl ContextEncoder for ScriptedEncoder {
    fn encode(&self, context: &Context) -> Result<ContextIds, AudioError> {
        Ok(ContextIds::new(
            context
                .keywords
                .iter()
                .enumerate()
                .map(|(i, _)| TokenId::new(1000 + u32::try_from(i).expect("small")))
                .collect(),
        ))
    }

    fn max_ids(&self) -> usize {
        4
    }
}

/// A recognizer frame (0.08 s) at or after mel frame `mel` (0.01 s).
fn frame_at(mel: usize) -> u32 {
    u32::try_from(mel.div_ceil(8)).expect("small")
}

/// Logs `off:<label>` for each step it takes, and writes `▁a` at each
/// transcribed span's start and `▁b` at its middle, in absolute frames.
struct ScriptedOffline {
    label: String,
    spans: Vec<Span>,
    next: usize,
    batch: usize,
    log: Log,
}

impl OfflineJob for ScriptedOffline {
    // None: every transcribed span has been returned.
    fn step(&mut self, _mel: &MelFrames) -> Result<Option<OfflineBatch>, AudioError> {
        self.log
            .lock()
            .expect("unpoisoned")
            .push(format!("off:{}", self.label));
        std::thread::sleep(Duration::from_millis(2));
        let mut out = Vec::new();
        while self.next < self.spans.len() && out.len() < self.batch {
            let span = self.spans[self.next];
            self.next += 1;
            if !span.transcribe {
                continue;
            }
            out.push(SpanResult {
                span,
                tokens: vec![
                    TokenEvent {
                        token: TokenId::new(1),
                        frame: frame_at(span.start),
                    },
                    TokenEvent {
                        token: TokenId::new(2),
                        frame: frame_at(span.start.midpoint(span.end)),
                    },
                ],
                chunk_end: frame_at(span.end),
                language: Some("xx".into()),
            });
        }
        Ok((!out.is_empty()).then_some(OfflineBatch { spans: out }))
    }
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
        // A stream conditioned on `tail` ends on half a character, which
        // only the detokenizer's flush reports.
        let last = if self.label == "tail" { 3 } else { 1 };
        Ok(vec![TokenEvent {
            token: TokenId::new(last),
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
    fn batches_offline(&self) -> usize {
        self.offline
    }
    fn context_encoder(&self) -> Option<Arc<dyn ContextEncoder>> {
        self.encoder.clone()
    }
    fn open_offline(
        &self,
        options: &StreamOptions,
        spans: Vec<Span>,
    ) -> Result<Box<dyn OfflineJob>, AudioError> {
        self.opened
            .lock()
            .expect("unpoisoned")
            .push(options.clone());
        self.opened_offline
            .lock()
            .expect("unpoisoned")
            .push(spans.clone());
        let label = match &options.language {
            Language::Auto => "auto".to_owned(),
            Language::Tag(tag) => tag.clone(),
        };
        Ok(Box::new(ScriptedOffline {
            label,
            spans,
            next: 0,
            batch: self.offline,
            log: Arc::clone(&self.log),
        }))
    }
}

/// Who speaks when in a scripted diarizer's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pattern {
    /// Speaker 0 for the first half, speaker 1 for the rest.
    Halves,
    /// Nobody, anywhere.
    Silent,
    /// Speaker 0 for the first third, nobody for the second, speaker 1 for
    /// the last.
    Gapped,
    /// No answer: the pass fails.
    Fails,
}

impl Pattern {
    /// The two speakers' activity at frame `f` of `n`.
    fn at(self, f: usize, n: usize) -> [f32; 2] {
        let first = match self {
            Self::Halves => f < n / 2,
            Self::Silent | Self::Fails => false,
            Self::Gapped => f < n / 3,
        };
        let second = match self {
            Self::Halves => f >= n / 2,
            Self::Silent | Self::Fails => false,
            Self::Gapped => f >= 2 * n / 3,
        };
        [
            if first { 0.9 } else { 0.0 },
            if second { 0.9 } else { 0.0 },
        ]
    }

    /// The answer over `n` frames.
    fn activity(self, n: usize) -> SpeakerActivity {
        let probs = (0..n).flat_map(|f| self.at(f, n)).collect();
        SpeakerActivity::new(0.01, 2, probs).expect("geometry")
    }
}

/// Answers its [`Pattern`] after `steps` logged steps of 5 ms each: long
/// enough that recognition running beside it shows in the log.
#[derive(Debug)]
struct Halves {
    spec: MelSpec,
    log: Log,
    pattern: Pattern,
    steps: usize,
}

struct HalvesPass {
    frames: usize,
    left: usize,
    log: Log,
    pattern: Pattern,
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
        if self.pattern == Pattern::Fails {
            return Err(AudioError::BadSpec {
                reason: "scripted diarizer failure".into(),
            });
        }
        Ok(Some(self.pattern.activity(self.frames)))
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
            left: self.steps,
            log: Arc::clone(&self.log),
            pattern: self.pattern,
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
        ..ServiceShape::plain(log)
    })
}

/// What a test service is built with.
#[expect(
    clippy::struct_excessive_bools,
    reason = "each names an independent knob a test flips on its own; a state \
              machine over them would be harder to read at every call site"
)]
struct ServiceShape<'a> {
    max_streams: usize,
    queue: usize,
    log: &'a Log,
    max_upload_bytes: usize,
    chunks: bool,
    opened: &'a Opened,
    /// A byte-level vocabulary of the same ids, whose pieces keep their
    /// leading space and whose id 3 is half a character.
    byte_level: bool,
    /// Spans an offline step takes; 0 is no offline path.
    offline: usize,
    opened_offline: &'a OpenedOffline,
    /// The diarizer's answer, or no diarizer.
    diarizer: Option<Pattern>,
    offline_min_secs: u32,
    skip_silence: bool,
    /// Steps of 5 ms the diarizer's pass takes.
    diar_steps: usize,
    /// Whether the recognizer hands the server a [`ScriptedEncoder`].
    encoder: bool,
}

impl<'a> ServiceShape<'a> {
    /// One job at a time, a queue of four, the halves diarizer, and no
    /// offline path.
    fn plain(log: &'a Log) -> Self {
        Self {
            max_streams: 1,
            queue: 4,
            log,
            max_upload_bytes: super::DEFAULT_MAX_UPLOAD_BYTES,
            chunks: false,
            opened: leaked_default(),
            byte_level: false,
            offline: 0,
            opened_offline: leaked_default(),
            diarizer: Some(Pattern::Halves),
            offline_min_secs: super::DEFAULT_OFFLINE_MIN_SECS,
            skip_silence: true,
            diar_steps: 20,
            encoder: false,
        }
    }

    /// A chunking recognizer that batches two spans a step and sends every
    /// non-streaming job offline, recording what it opens.
    fn offline(
        log: &'a Log,
        opened: &'a Opened,
        opened_offline: &'a OpenedOffline,
        diarizer: Option<Pattern>,
    ) -> Self {
        Self {
            max_streams: 2,
            chunks: true,
            opened,
            offline: 2,
            opened_offline,
            diarizer,
            offline_min_secs: 0,
            ..Self::plain(log)
        }
    }
}

/// A record nobody reads, for a shape that does not look at what it opens.
fn leaked_default<T: Default>() -> &'static T {
    Box::leak(Box::default())
}

fn service_with(shape: &ServiceShape<'_>) -> AudioService {
    let ServiceShape {
        max_streams,
        queue,
        log,
        max_upload_bytes,
        chunks,
        opened,
        byte_level,
        offline,
        opened_offline,
        diarizer,
        offline_min_secs,
        skip_silence,
        diar_steps,
        encoder,
    } = *shape;
    let vocab = Arc::new(if byte_level {
        PieceVocab::from_byte_level(
            vec![
                "!".into(),
                "\u{120}a".into(),
                "\u{120}b".into(),
                "\u{e5}".into(),
            ],
            &[],
        )
        .expect("pieces")
    } else {
        PieceVocab::from_pieces(vec![
            "<blank>".into(),
            "\u{2581}a".into(),
            "\u{2581}b".into(),
        ])
        .expect("pieces")
    });
    let (service, _join) = AudioService::start(AudioServiceDesc {
        recognizer: Arc::new(Scripted {
            spec: spec(),
            vocab,
            log: Arc::clone(log),
            chunks,
            opened: Arc::clone(opened),
            offline,
            opened_offline: Arc::clone(opened_offline),
            encoder: encoder.then(|| -> Arc<dyn ContextEncoder> { Arc::new(ScriptedEncoder) }),
        }),
        diarizer: diarizer.map(|pattern| -> Arc<dyn Diarizer> {
            Arc::new(Halves {
                spec: spec(),
                log: Arc::clone(log),
                pattern,
                steps: diar_steps,
            })
        }),
        asr_id: "scripted".into(),
        lookahead: 0,
        max_streams,
        queue,
        limits: crate::tokenizer::message_preprocess::DEFAULT_AUDIO_LIMITS,
        ffmpeg: None,
        max_upload_bytes,
        offline_min_secs,
        skip_silence,
        max_context_ids: 3,
    })
    .expect("the thread starts");
    service
}

fn job(seconds: usize, language: &str, diarize: bool) -> TranscriptionJob {
    TranscriptionJob {
        pcm: pcm_secs_usize(seconds),
        language: Language::Tag(language.into()),
        context: ContextIds::default(),
        diarize,
        streaming: false,
    }
}

/// The same synthetic clip [`job`] builds, `seconds` long.
fn pcm_secs_usize(seconds: usize) -> Pcm {
    let samples = (0..seconds * 16_000)
        .map(|i| (f32::from(u16::try_from(i % 97).expect("small")) / 97.0) - 0.5)
        .collect();
    Pcm::new(samples, 16_000).expect("pcm")
}

/// [`pcm_secs_usize`], for a fractional clip length.
#[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "a test clip's length in seconds is always small and non-negative"
)]
fn pcm_secs(seconds: f64) -> Pcm {
    pcm_secs_usize(seconds.round() as usize)
}

/// [`job`], read as it streams (`stream=true`).
fn streamed_job(seconds: usize, language: &str) -> TranscriptionJob {
    TranscriptionJob {
        streaming: true,
        ..job(seconds, language, false)
    }
}

/// The recognizer's mel of `clip`, as the worker frames it.
fn mel_of(clip: &TranscriptionJob) -> MelFrames {
    let all = MelFrontend::new(spec())
        .expect("spec")
        .compute(clip.pcm.samples());
    let (n, valid) = (all.n_mels(), all.valid());
    MelFrames::from_rows(all.as_slice()[..valid * n].to_vec(), n).expect("rows")
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
async fn a_transcript_is_trimmed_and_flushed_at_its_end() {
    let log = Log::default();
    let svc = service_with(&ServiceShape {
        max_streams: 2,
        byte_level: true,
        ..ServiceShape::plain(&log)
    });
    let events = drain(svc.submit(job(1, "tail", false)).expect("queued")).await;
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
    // Byte-level pieces keep the space before the first word; the text
    // does not.
    assert!(deltas.starts_with(" a"), "{deltas:?}");
    assert!(transcript.text.starts_with("a b"), "{:?}", transcript.text);
    // The half character at the end is reported, in the text and as a
    // delta, rather than dropped.
    assert!(
        transcript.text.ends_with('\u{FFFD}'),
        "{:?}",
        transcript.text
    );
    assert_eq!(deltas.trim(), transcript.text);
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
        chunks: true,
        opened: &opened,
        ..ServiceShape::plain(&log)
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
        opened: &opened,
        ..ServiceShape::plain(&log)
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
    // A second of diarizing: admitting the plain job computes its mel first,
    // which a busy test machine can stretch past a shorter pass.
    let svc = service_with(&ServiceShape {
        max_streams: 2,
        diar_steps: 200,
        ..ServiceShape::plain(&log)
    });
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
) -> (StatusCode, serde_json::Value) {
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
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
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
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(body["error"]["code"], "request_too_large", "{body}");
}

/// The offline steps the log shows, in order: `a` for `off:a`.
fn steps(log: &Log) -> Vec<String> {
    log.lock()
        .expect("unpoisoned")
        .iter()
        .filter_map(|l| l.strip_prefix("off:").map(str::to_owned))
        .collect()
}

fn transcribed(plan: &[Span]) -> usize {
    plan.iter().filter(|s| s.transcribe).count()
}

#[tokio::test]
async fn an_offline_job_steps_one_batch_per_turn() {
    let (log, opened, offline) = (Log::default(), Opened::default(), OpenedOffline::default());
    let svc = service_with(&ServiceShape::offline(&log, &opened, &offline, None));
    // Long enough for several spans each under the silence rule.
    let a = svc.submit(job(100, "a", false)).expect("queued");
    let b = svc.submit(job(100, "b", false)).expect("queued");
    let (a, b) = tokio::join!(drain(a), drain(b));
    let Some(TranscriptEvent::Done(done)) = a.last() else {
        panic!("{a:?}");
    };
    assert!(matches!(b.last(), Some(TranscriptEvent::Done(_))), "{b:?}");
    // `opened` now records both `open_stream` and `open_offline` calls, so
    // the stream count is what is left once the offline opens are taken out;
    // here that is zero, since both jobs went offline.
    let opened_len = opened.lock().expect("unpoisoned").len();
    let offline_len = offline.lock().expect("unpoisoned").len();
    assert_eq!(
        opened_len - offline_len,
        0,
        "no RecognizerStream::open_stream call: both jobs went offline \
         (opened: {opened_len}, offline: {offline_len})"
    );
    let plans = offline.lock().expect("unpoisoned").clone();
    assert_eq!(plans.len(), 2);
    let spans = transcribed(&plans[0]);
    assert!(spans >= 3, "{:?}", plans[0]);
    // Every span wrote its two words, in plan order.
    assert_eq!(done.words.len(), 2 * spans, "{:?}", done.words);
    assert_eq!(done.language.as_deref(), Some("xx"));
    let steps = steps(&log);
    // Two spans a step, then the step that says the job is done.
    for label in ["a", "b"] {
        let taken = steps.iter().filter(|s| *s == label).count();
        assert_eq!(taken, spans.div_ceil(2) + 1, "{label}: {steps:?}");
    }
    // One step a turn: while both run, their steps alternate.
    let first_b = steps.iter().position(|s| s == "b").expect("b stepped");
    let last_a = steps.iter().rposition(|s| s == "a").expect("a stepped");
    assert!(first_b < last_a, "a finished before b stepped: {steps:?}");
    assert!(
        steps[first_b..=last_a].windows(2).all(|w| w[0] != w[1]),
        "a job took two steps in one turn: {steps:?}"
    );
}

#[tokio::test]
async fn a_streaming_job_is_not_held_behind_an_offline_job() {
    let (log, opened, offline) = (Log::default(), Opened::default(), OpenedOffline::default());
    let svc = service_with(&ServiceShape::offline(&log, &opened, &offline, None));
    let batched = svc.submit(job(100, "a", false)).expect("queued");
    let live = svc.submit(streamed_job(40, "b")).expect("queued");
    let (batched, live) = tokio::join!(drain(batched), drain(live));
    assert!(matches!(batched.last(), Some(TranscriptEvent::Done(_))));
    assert!(matches!(live.last(), Some(TranscriptEvent::Done(_))));
    // `opened` now records both `open_stream` and `open_offline` calls, so
    // the stream count is what is left once the offline opens are taken
    // out: one, for the streaming job; the batched job went offline instead.
    let opened_len = opened.lock().expect("unpoisoned").len();
    let offline_len = offline.lock().expect("unpoisoned").len();
    assert_eq!(
        opened_len - offline_len,
        1,
        "one real stream open: the live job (opened: {opened_len}, offline: {offline_len})"
    );
    assert_eq!(offline_len, 1);
    let log = log.lock().expect("unpoisoned").clone();
    let first_b = log.iter().position(|l| l == "b").expect("b ran");
    let last_a = log.iter().rposition(|l| l == "off:a").expect("a stepped");
    assert!(
        first_b < last_a,
        "the streaming job waited for the offline one: {log:?}"
    );
}

#[tokio::test]
async fn a_streaming_job_beats_two_offline_jobs_to_its_first_delta() {
    let (log, opened, offline) = (Log::default(), Opened::default(), OpenedOffline::default());
    let svc = service_with(&ServiceShape {
        // All three admitted at once: two offline jobs and a streaming one
        // beside them, so none waits in the queue for a slot.
        max_streams: 3,
        ..ServiceShape::offline(&log, &opened, &offline, None)
    });
    let mut a = svc.submit(job(100, "a", false)).expect("queued");
    let mut b = svc.submit(job(100, "b", false)).expect("queued");
    let mut live = svc.submit(streamed_job(40, "c")).expect("queued");

    // Whether each offline job had already sent its last event when the
    // streaming job's first delta arrived, and each job's own last event -
    // raced across the three channels rather than drained one at a time, so
    // the order the worker actually delivered them in is what is checked.
    let mut a_last = None;
    let mut b_last = None;
    let mut live_last = None;
    let mut delta_beat_both_offline_jobs = None;
    tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            tokio::select! {
                Some(event) = a.recv(), if a_last.is_none() => {
                    if matches!(event, TranscriptEvent::Done(_) | TranscriptEvent::Failed(_)) {
                        a_last = Some(event);
                    }
                }
                Some(event) = b.recv(), if b_last.is_none() => {
                    if matches!(event, TranscriptEvent::Done(_) | TranscriptEvent::Failed(_)) {
                        b_last = Some(event);
                    }
                }
                Some(event) = live.recv(), if live_last.is_none() => {
                    if delta_beat_both_offline_jobs.is_none()
                        && matches!(event, TranscriptEvent::Delta(_))
                    {
                        delta_beat_both_offline_jobs =
                            Some(a_last.is_none() && b_last.is_none());
                    }
                    if matches!(event, TranscriptEvent::Done(_) | TranscriptEvent::Failed(_)) {
                        live_last = Some(event);
                    }
                }
                else => break,
            }
        }
    })
    .await
    .expect("the worker answers");

    assert!(
        matches!(a_last, Some(TranscriptEvent::Done(_))),
        "{a_last:?}"
    );
    assert!(
        matches!(b_last, Some(TranscriptEvent::Done(_))),
        "{b_last:?}"
    );
    assert!(
        matches!(live_last, Some(TranscriptEvent::Done(_))),
        "{live_last:?}"
    );
    assert_eq!(
        delta_beat_both_offline_jobs,
        Some(true),
        "the streaming job's first delta should arrive while both offline jobs are still running"
    );
}

#[tokio::test]
async fn an_all_silent_diarized_job_completes_empty() {
    let (log, opened, offline) = (Log::default(), Opened::default(), OpenedOffline::default());
    let svc = service_with(&ServiceShape::offline(
        &log,
        &opened,
        &offline,
        Some(Pattern::Silent),
    ));
    let clip = job(6, "a", true);
    let frames = mel_of(&clip).valid();
    let events = drain(svc.submit(clip).expect("queued")).await;
    let Some(TranscriptEvent::Done(done)) = events.last() else {
        panic!("{events:?}");
    };
    assert_eq!(done.text, "");
    assert!(
        done.words.is_empty() && done.segments.is_empty(),
        "{done:?}"
    );
    assert!(done.diarized);
    assert_eq!(
        offline.lock().expect("unpoisoned").clone(),
        vec![vec![Span {
            start: 0,
            end: frames,
            transcribe: false,
        }]]
    );
}

#[tokio::test]
async fn a_job_without_a_diarizer_uses_the_silence_rule() {
    let (log, opened, offline) = (Log::default(), Opened::default(), OpenedOffline::default());
    let svc = service_with(&ServiceShape::offline(&log, &opened, &offline, None));
    let clip = job(70, "a", false);
    let want = chunks::plan_offline(&mel_of(&clip), None, 0.01, &OfflineRule::for_10ms_mel(true))
        .expect("plans");
    assert!(
        want.len() >= 2 && want.iter().all(|s| s.transcribe),
        "{want:?}"
    );
    let events = drain(svc.submit(clip).expect("queued")).await;
    let Some(TranscriptEvent::Done(done)) = events.last() else {
        panic!("{events:?}");
    };
    assert!(!done.diarized);
    assert_eq!(offline.lock().expect("unpoisoned").clone(), vec![want]);
}

#[tokio::test]
async fn a_plain_job_is_cut_at_the_diarizer_turns_without_naming_speakers() {
    let (log, opened, offline) = (Log::default(), Opened::default(), OpenedOffline::default());
    let svc = service_with(&ServiceShape::offline(
        &log,
        &opened,
        &offline,
        Some(Pattern::Halves),
    ));
    let clip = job(6, "a", false);
    let mel = mel_of(&clip);
    let activity = Pattern::Halves.activity(mel.valid());
    let want = chunks::plan_offline(
        &mel,
        Some(&activity),
        0.01,
        &OfflineRule::for_10ms_mel(true),
    )
    .expect("plans");
    assert_eq!(want.len(), 2, "one turn change: {want:?}");
    let events = drain(svc.submit(clip).expect("queued")).await;
    let Some(TranscriptEvent::Done(done)) = events.last() else {
        panic!("{events:?}");
    };
    assert!(!done.diarized);
    assert!(done.segments.iter().all(|s| s.speaker.is_none()));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, TranscriptEvent::Segment(_))),
        "{events:?}"
    );
    assert_eq!(offline.lock().expect("unpoisoned").clone(), vec![want]);
}

#[tokio::test]
async fn a_plain_offline_job_whose_diarizer_fails_is_cut_by_the_silence_rule() {
    let (log, opened, offline) = (Log::default(), Opened::default(), OpenedOffline::default());
    let svc = service_with(&ServiceShape::offline(
        &log,
        &opened,
        &offline,
        Some(Pattern::Fails),
    ));
    let clip = job(70, "a", false);
    let want = chunks::plan_offline(&mel_of(&clip), None, 0.01, &OfflineRule::for_10ms_mel(true))
        .expect("plans");
    assert!(want.len() >= 2, "{want:?}");
    let events = drain(svc.submit(clip).expect("queued")).await;
    let Some(TranscriptEvent::Done(done)) = events.last() else {
        panic!("the transcript completes without the diarizer: {events:?}");
    };
    assert!(!done.diarized);
    assert!(!done.text.is_empty(), "{done:?}");
    assert_eq!(offline.lock().expect("unpoisoned").clone(), vec![want]);
}

#[tokio::test]
async fn a_diarized_offline_job_whose_diarizer_fails_fails() {
    let (log, opened, offline) = (Log::default(), Opened::default(), OpenedOffline::default());
    let svc = service_with(&ServiceShape::offline(
        &log,
        &opened,
        &offline,
        Some(Pattern::Fails),
    ));
    let events = drain(svc.submit(job(70, "a", true)).expect("queued")).await;
    let Some(TranscriptEvent::Failed(failure)) = events.last() else {
        panic!("the speakers it asked for cannot be named: {events:?}");
    };
    assert!(!failure.caller, "{failure:?}");
    assert!(
        failure.message.contains("scripted diarizer failure"),
        "{failure:?}"
    );
    assert!(
        offline.lock().expect("unpoisoned").is_empty(),
        "no offline job opens"
    );
}

#[tokio::test]
async fn skipped_spans_produce_no_segments_and_frames_stay_absolute() {
    let (log, opened, offline) = (Log::default(), Opened::default(), OpenedOffline::default());
    let svc = service_with(&ServiceShape::offline(
        &log,
        &opened,
        &offline,
        Some(Pattern::Gapped),
    ));
    let clip = job(12, "a", true);
    let mel = mel_of(&clip);
    let activity = Pattern::Gapped.activity(mel.valid());
    let want = chunks::plan_offline(
        &mel,
        Some(&activity),
        0.01,
        &OfflineRule::for_10ms_mel(true),
    )
    .expect("plans");
    let skipped: Vec<Span> = want.iter().filter(|s| !s.transcribe).copied().collect();
    assert_eq!(skipped.len(), 1, "the middle third is skipped: {want:?}");
    let gap = skipped[0];
    let secs = |frame: usize| f64::from(u32::try_from(frame).expect("small")) * 0.01;
    let (gap_start, gap_end) = (secs(gap.start), secs(gap.end));
    let events = drain(svc.submit(clip).expect("queued")).await;
    let Some(TranscriptEvent::Done(done)) = events.last() else {
        panic!("{events:?}");
    };
    assert_eq!(offline.lock().expect("unpoisoned").clone(), vec![want]);
    // Two words for each side of the gap, none inside it.
    assert_eq!(done.words.len(), 4, "{:?}", done.words);
    assert!(
        done.words
            .iter()
            .all(|w| w.start < gap_start || w.start >= gap_end),
        "{:?}",
        done.words
    );
    // The words after the gap sit after it, not at the clip's start.
    assert!(done.words[2].start >= gap_end, "{:?}", done.words);
    // One recognizer frame (0.08 s) of rounding at the gap's edges.
    assert!(
        done.segments
            .iter()
            .all(|s| s.end <= gap_start + 0.08 || s.start >= gap_end),
        "{:?}",
        done.segments
    );
    let speakers: Vec<Option<usize>> = done.segments.iter().map(|s| s.speaker).collect();
    assert_eq!(speakers.first(), Some(&Some(0)), "{speakers:?}");
    assert_eq!(speakers.last(), Some(&Some(1)), "{speakers:?}");
}

#[tokio::test]
async fn a_clip_under_the_offline_threshold_keeps_the_chunked_stream_path() {
    for (min_secs, offline_jobs) in [(super::DEFAULT_OFFLINE_MIN_SECS, 0), (0, 1)] {
        let (log, opened, offline) = (Log::default(), Opened::default(), OpenedOffline::default());
        let svc = service_with(&ServiceShape {
            offline_min_secs: min_secs,
            ..ServiceShape::offline(&log, &opened, &offline, None)
        });
        let events = drain(svc.submit(job(240, "a", false)).expect("queued")).await;
        assert!(matches!(events.last(), Some(TranscriptEvent::Done(_))));
        assert_eq!(
            offline.lock().expect("unpoisoned").len(),
            offline_jobs,
            "threshold {min_secs}"
        );
        // `opened` records both `open_stream` and `open_offline` calls, so
        // the stream count is what is left once the offline opens are taken
        // out: one below the threshold (a real stream), zero at or above it
        // (the job went offline instead).
        let opened_len = opened.lock().expect("unpoisoned").len();
        let offline_len = offline.lock().expect("unpoisoned").len();
        assert_eq!(
            opened_len - offline_len,
            1 - offline_jobs,
            "threshold {min_secs} (opened: {opened_len}, offline: {offline_len})"
        );
    }
}

#[tokio::test]
async fn a_context_reaches_every_stream_the_job_opens() {
    let log = Log::default();
    let opened = Opened::default();
    let offline = OpenedOffline::default();
    let service = service_with(&ServiceShape {
        encoder: true,
        ..ServiceShape::offline(&log, &opened, &offline, None)
    });
    let context = Context {
        prompt: None,
        keywords: vec!["a".into(), "b".into()],
    };
    let ids = service.encode_context(&context).expect("encodes");
    assert_eq!(ids.len(), 2);
    for streaming in [true, false] {
        let events = service
            .submit(TranscriptionJob {
                pcm: pcm_secs(2.0),
                language: Language::Auto,
                context: ids.clone(),
                diarize: false,
                streaming,
            })
            .expect("queued");
        drain(events).await;
    }
    let seen = opened.lock().expect("unpoisoned");
    assert_eq!(seen.len(), 2, "one stream and one offline job");
    assert!(seen.iter().all(|options| options.context == ids));
}

/// A diarized job on a chunking recognizer waits in `Stream::AwaitingTurns`
/// until the diarizer's turns are known, then opens through
/// `worker::open_at_turns` rather than the immediate `open_stream` call in
/// `worker::start`; its context must survive that wait too.
#[tokio::test]
async fn a_context_reaches_a_stream_opened_at_the_diarizer_turns() {
    let log = Log::default();
    let opened = Opened::default();
    // `ServiceShape::plain` already loads a diarizer (`Pattern::Halves`);
    // `chunks: true` is what sends a diarized job through `AwaitingTurns`
    // instead of opening its stream immediately.
    let service = service_with(&ServiceShape {
        chunks: true,
        opened: &opened,
        encoder: true,
        ..ServiceShape::plain(&log)
    });
    let context = Context {
        prompt: None,
        keywords: vec!["a".into()],
    };
    let ids = service.encode_context(&context).expect("encodes");
    let events = drain(
        service
            .submit(TranscriptionJob {
                context: ids.clone(),
                ..job(6, "a", true)
            })
            .expect("queued"),
    )
    .await;
    assert!(matches!(events.last(), Some(TranscriptEvent::Done(_))));
    let seen = opened.lock().expect("unpoisoned").clone();
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0].context, ids);
}

#[test]
fn a_context_is_refused_by_a_recognizer_that_takes_none() {
    let log = Log::default();
    let service = service_with(&ServiceShape::plain(&log));
    let error = service
        .encode_context(&Context {
            prompt: Some("x".into()),
            keywords: Vec::new(),
        })
        .expect_err("refused");
    assert_eq!(error.status(), StatusCode::BAD_REQUEST, "{error}");
}

#[test]
fn a_context_past_the_server_cap_is_refused() {
    let log = Log::default();
    let service = service_with(&ServiceShape {
        encoder: true,
        ..ServiceShape::plain(&log)
    });
    let error = service
        .encode_context(&Context {
            prompt: None,
            keywords: vec!["a".into(), "b".into(), "c".into(), "d".into()],
        })
        .expect_err("refused");
    assert_eq!(error.status(), StatusCode::BAD_REQUEST, "{error}");
    assert!(
        error.to_string().contains("--asr-max-context-tokens"),
        "{error}"
    );
}

#[test]
fn a_context_past_the_byte_cap_is_refused_before_tokenizing() {
    let log = Log::default();
    let service = service_with(&ServiceShape {
        encoder: true,
        ..ServiceShape::plain(&log)
    });
    // A long prompt and no keywords: the scripted encoder would emit no ids
    // for it (it only counts keywords), so a refusal here can only come from
    // the byte check, not the ids-per-request cap.
    let error = service
        .encode_context(&Context {
            prompt: Some("x".repeat(super::MAX_CONTEXT_BYTES + 1)),
            keywords: Vec::new(),
        })
        .expect_err("refused");
    assert_eq!(error.status(), StatusCode::BAD_REQUEST, "{error}");
    assert!(
        !error.to_string().contains("--asr-max-context-tokens"),
        "the byte cap is refused before an id count exists: {error}"
    );
}
