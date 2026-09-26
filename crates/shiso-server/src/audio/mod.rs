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

mod diarize;
mod worker;

use std::sync::Arc;
use std::thread::JoinHandle;

use shiso_audio::transcript::{Segment, Word};
use shiso_audio::{AudioError, DecodeLimits, Diarizer, Language, Pcm, SpeechRecognizer};
use shiso_vision::FfmpegPolicy;
use tokio::sync::mpsc;

use crate::error::AppError;

/// One transcription to run.
#[derive(Debug)]
pub struct TranscriptionJob {
    /// The audio, already at the model rate.
    pub pcm: Pcm,
    /// Language conditioning.
    pub language: Language,
    /// Whether to attribute each word to a speaker.
    pub diarize: bool,
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
    /// The subprocess decoder for containers other than WAV, or `None` when
    /// only WAV is accepted.
    pub ffmpeg: Option<FfmpegPolicy>,
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
    ffmpeg: Option<Arc<FfmpegPolicy>>,
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
            ffmpeg: desc.ffmpeg.clone().map(Arc::new),
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

    /// Whether a diarizer is loaded.
    #[must_use]
    pub const fn has_diarizer(&self) -> bool {
        self.diarizer
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
        let ffmpeg = self.ffmpeg.clone();
        tokio::task::spawn_blocking(move || {
            shiso_audio::decode_audio(&bytes, &limits, ffmpeg.as_deref())
        })
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
