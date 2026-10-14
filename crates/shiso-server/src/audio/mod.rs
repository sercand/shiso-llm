//! Transcription: the service the HTTP handlers submit to, and the thread
//! that runs it.
//!
//! The speech models run on a thread of their own, over a backend instance of
//! their own, beside the engine thread: a transcription job and a chat turn
//! must not share a command stream, because a device that records one stream
//! for replay (graph capture) would record the other's launches into it.
//!
//! Jobs are admitted from a bounded queue and advanced round robin, one mel
//! chunk each per turn, so a long file does not hold a short one behind it.
//! Each job streams its events back on its own channel; a caller that stops
//! listening frees its slot at the job's next event. A diarized job's
//! speaker pass runs on a third thread, over a third backend instance, while
//! its words are recognised.
//!
//! A long clip nobody reads as it streams goes another way on a recognizer
//! that batches ([`SpeechRecognizer::batches_offline`]): it is planned whole
//! into spans - at the diarizer's turns when one is loaded, with long
//! silences skipped - and transcribed a batch of spans a turn.

mod diarize;
mod worker;

use std::sync::Arc;
use std::thread::JoinHandle;

use shiso_audio::transcript::{Segment, Word};
use shiso_audio::{
    AudioError, Context, ContextEncoder, ContextIds, DecodeLimits, Diarizer, Language, Pcm,
    SpeechRecognizer,
};
use tokio::sync::mpsc;

use crate::error::AppError;

/// The most raw bytes a context's prompt and keywords may total together,
/// checked before either is tokenized. The upload cap is hundreds of MB and
/// this runs on the caller's (async runtime) thread, so a huge `prompt`
/// would otherwise tokenize synchronously there; the largest ids cap any
/// recognizer offers is 512, and 512 ids never need more than a few KiB of
/// source text.
const MAX_CONTEXT_BYTES: usize = 16 * 1024;

/// The largest transcription upload, when nothing else is configured:
/// 512 MiB, a little over an hour of 44.1 kHz stereo WAV and several hours of
/// compressed audio. The whole body is held while it is decoded, so this is
/// also the memory one request may pin.
pub const DEFAULT_MAX_UPLOAD_BYTES: usize = 512 * 1024 * 1024;

/// The shortest clip, in seconds, that goes offline when nothing else is
/// configured: 5 minutes. A shorter clip keeps the chunk-by-chunk path, whose
/// first text arrives a chunk in and whose whole run is short anyway.
pub const DEFAULT_OFFLINE_MIN_SECS: u32 = 300;

/// One transcription to run.
#[derive(Debug)]
pub struct TranscriptionJob {
    /// The audio, already at the model rate.
    pub pcm: Pcm,
    /// Language conditioning.
    pub language: Language,
    /// The prompt's context, as the recognizer's ids
    /// ([`AudioService::encode_context`]); empty for none.
    pub context: ContextIds,
    /// Whether to attribute each word to a speaker.
    pub diarize: bool,
    /// Whether the caller reads the events as they come (`stream=true`).
    /// Such a job keeps the chunk-by-chunk path, whose first text arrives a
    /// chunk in; an offline job's arrives a batch in.
    pub streaming: bool,
}

/// What a job reports, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptEvent {
    /// Text appended to the transcript.
    Delta(String),
    /// A speaker segment that will not change, for a diarized job.
    Segment(Segment),
    /// The finished transcript; the last event of a job that succeeded.
    Done(Transcript),
    /// Why the job stopped; the last event of a job that failed.
    Failed(JobFailure),
}

/// Why a job stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobFailure {
    /// What went wrong, written for whoever sent the audio.
    pub message: String,
    /// Whether the request was at fault (audio too short, a language the
    /// model does not know) rather than this server.
    pub caller: bool,
}

/// A finished transcript.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    /// The whole text.
    pub text: String,
    /// The language a tag in the output reported, when the model emits one.
    pub language: Option<String>,
    /// Seconds of audio.
    pub duration: f64,
    /// Words with their times.
    pub words: Vec<Word>,
    /// Segments; each carries a speaker when the job was diarized.
    pub segments: Vec<Segment>,
    /// Whether the segments carry speakers.
    pub diarized: bool,
}

/// What the worker thread is started with.
#[derive(Debug)]
pub struct AudioServiceDesc {
    /// The recognizer, loaded on the audio backend.
    pub recognizer: Arc<dyn SpeechRecognizer>,
    /// The diarizer, when one was configured, loaded on a backend instance
    /// of its own: it runs on its own thread, beside the recognizer.
    pub diarizer: Option<Arc<dyn Diarizer>>,
    /// The name the recognizer answers to.
    pub asr_id: String,
    /// Encoder lookahead every stream runs at.
    pub lookahead: usize,
    /// Jobs advanced at once.
    pub max_streams: usize,
    /// Jobs that may wait beyond those.
    pub queue: usize,
    /// The decode cap a request's audio is held to.
    pub limits: DecodeLimits,
    /// How containers other than WAV are decoded.
    pub decode: shiso_audio::AudioDecodePolicy,
    /// The largest request body the transcription routes read.
    pub max_upload_bytes: usize,
    /// The shortest decoded clip, in seconds, that a non-streaming job
    /// transcribes offline on a recognizer that batches; 0 sends every one.
    pub offline_min_secs: u32,
    /// Whether an offline job skips the long silences its diarizer finds.
    pub skip_silence: bool,
    /// The most context ids a request may carry
    /// (`--asr-max-context-tokens`), at most the recognizer's own cap.
    pub max_context_ids: usize,
}

/// A queued job and where its events go.
#[derive(Debug)]
struct Queued {
    job: TranscriptionJob,
    events: mpsc::UnboundedSender<TranscriptEvent>,
}

/// The handle the HTTP handlers hold.
#[derive(Debug, Clone)]
pub struct AudioService {
    tx: mpsc::Sender<Queued>,
    asr_id: String,
    languages: Arc<[String]>,
    diarizer: bool,
    limits: DecodeLimits,
    decode: Arc<shiso_audio::AudioDecodePolicy>,
    max_upload_bytes: usize,
    /// The encoder the recognizer hands the server for its context, or
    /// `None` for a recognizer that takes none.
    context: Option<Arc<dyn ContextEncoder>>,
    max_context_ids: usize,
}

impl AudioService {
    /// Starts the worker thread.
    ///
    /// # Errors
    ///
    /// The operating system's refusal to start a thread.
    pub fn start(desc: AudioServiceDesc) -> std::io::Result<(Self, JoinHandle<()>)> {
        let (tx, rx) = mpsc::channel(desc.queue.max(1));
        let service = Self {
            tx,
            asr_id: desc.asr_id.clone(),
            languages: desc.recognizer.languages().into(),
            diarizer: desc.diarizer.is_some(),
            limits: desc.limits,
            decode: Arc::new(desc.decode.clone()),
            max_upload_bytes: desc.max_upload_bytes,
            context: desc.recognizer.context_encoder(),
            max_context_ids: desc.max_context_ids,
        };
        // The diarizer thread ends when the worker, its only sender, does.
        let diarizer = match &desc.diarizer {
            Some(model) => {
                let (queue, requests) = std::sync::mpsc::channel();
                let model = Arc::clone(model);
                std::thread::Builder::new()
                    .name("shiso-diarize".to_owned())
                    .spawn(move || diarize::run(&*model, &requests))?;
                Some(queue)
            }
            None => None,
        };
        let join = std::thread::Builder::new()
            .name("shiso-audio".to_owned())
            .spawn(move || worker::run(&desc, rx, &diarizer))?;
        Ok((service, join))
    }

    /// The name the recognizer answers to.
    #[must_use]
    pub fn asr_id(&self) -> &str {
        &self.asr_id
    }

    /// Locale tags the recognizer's language conditioning accepts.
    #[must_use]
    pub fn languages(&self) -> &[String] {
        &self.languages
    }

    /// The largest request body the transcription routes read.
    #[must_use]
    pub const fn max_upload_bytes(&self) -> usize {
        self.max_upload_bytes
    }

    /// Whether a diarizer is loaded.
    #[must_use]
    pub const fn has_diarizer(&self) -> bool {
        self.diarizer
    }

    /// `context` as the recognizer's ids, checked against this server's
    /// cap. Runs on the caller's thread: the encoder is plain host data.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] naming `keywords` (or `prompt`, when the
    /// context carries no keywords) for a context sent to a recognizer that
    /// takes none, one whose raw prompt and keywords text totals more than
    /// 16 KiB, one longer than `--asr-max-context-tokens` once encoded (even
    /// when what the encoder hit first was its own, possibly looser, cap),
    /// or one the encoder refuses for another reason.
    pub fn encode_context(&self, context: &Context) -> Result<ContextIds, AppError> {
        if context.is_empty() {
            return Ok(ContextIds::default());
        }
        // A context with no keywords was refused over its `prompt`; one with
        // keywords (with or without a prompt too) was refused over those.
        let param = if context.keywords.is_empty() {
            "prompt"
        } else {
            "keywords"
        };
        let refuse =
            |message: String| AppError::bad_request(message, Some(param), "invalid_request");
        let encoder = self.context.as_ref().ok_or_else(|| {
            refuse(format!(
                "{} takes no keywords or prompt; send the audio alone",
                self.asr_id
            ))
        })?;
        // Checked before the encoder ever sees the text: the upload cap is
        // hundreds of MB and this runs on the async runtime, so a caller's
        // oversized prompt must not reach a tokenizer pass here.
        let bytes = context.prompt.as_deref().map_or(0, str::len)
            + context.keywords.iter().map(String::len).sum::<usize>();
        if bytes > MAX_CONTEXT_BYTES {
            return Err(refuse(format!(
                "the keywords and prompt are {bytes} bytes of text; this server encodes at \
                 most {MAX_CONTEXT_BYTES} bytes"
            )));
        }
        let too_many_tokens = |tokens: usize| {
            refuse(format!(
                "the keywords and prompt are {tokens} tokens; this server takes at most {} \
                 (--asr-max-context-tokens)",
                self.max_context_ids
            ))
        };
        let ids = encoder.encode(context).map_err(|e| match e {
            // The encoder's own cap can be looser than this server's
            // configured one, so its own message would quote a number an
            // operator never set. Restate the refusal in this server's own
            // terms instead of relaying the encoder's, keeping only the
            // token count, which is accurate regardless of whose cap it is
            // compared against.
            AudioError::TooManyContextTokens { tokens, .. } => too_many_tokens(tokens),
            other => refuse(other.to_string()),
        })?;
        if ids.len() > self.max_context_ids {
            return Err(too_many_tokens(ids.len()));
        }
        Ok(ids)
    }

    /// Decodes a request's audio to the model rate, off the async runtime: a
    /// WAV parse and a resample, or an ffmpeg run under its policy.
    ///
    /// # Errors
    ///
    /// The decoder's refusal, or [`AudioError::Io`] when the decode task
    /// could not run.
    pub async fn decode(&self, bytes: axum::body::Bytes) -> Result<Pcm, AudioError> {
        let limits = self.limits;
        let decode = Arc::clone(&self.decode);
        tokio::task::spawn_blocking(move || shiso_audio::decode_audio(&bytes, &limits, &decode))
            .await
            .map_err(|error| AudioError::Io {
                what: "running the audio decode",
                source: std::io::Error::other(error),
            })?
    }

    /// Queues `job` and returns the channel its events arrive on.
    ///
    /// # Errors
    ///
    /// [`AppError::Overloaded`] when the queue is full, and
    /// [`AppError::NotReady`] when the worker has stopped.
    pub fn submit(
        &self,
        job: TranscriptionJob,
    ) -> Result<mpsc::UnboundedReceiver<TranscriptEvent>, AppError> {
        let (events, rx) = mpsc::unbounded_channel();
        self.tx
            .try_send(Queued { job, events })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => AppError::Overloaded {
                    what: "the transcription queue is full",
                },
                mpsc::error::TrySendError::Closed(_) => AppError::NotReady {
                    what: "the transcription worker",
                },
            })?;
        Ok(rx)
    }
}

#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;
