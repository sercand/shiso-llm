//! The audio thread: admit, advance round robin, report.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::mpsc as std_mpsc;
use std::time::Duration;

use shiso_audio::chunks;
use shiso_audio::transcript::{self, SpeakerActivity, TimedPiece};
use shiso_audio::{
    AudioError, Detokenizer, Language, MelFrames, MelFrontend, RecognizerStream, SpeechRecognizer,
    StreamOptions, TokenEvent,
};
use tokio::sync::mpsc;

use super::diarize;
use super::{AudioServiceDesc, JobFailure, Queued, Transcript, TranscriptEvent, TranscriptionJob};

/// How long a turn waits for a recognised job's speakers before the round
/// moves on: short, so the other jobs barely notice, and long enough that a
/// lone job waiting on its diarizer is not a busy loop.
const SPEAKER_WAIT: Duration = Duration::from_millis(5);

/// Mel rows one round of turns shares among the active jobs: 16 s of audio.
/// A lone job takes all of it, because the encoder runs every chunk a push
/// completes in one pass and a wide pass is far cheaper per frame than many
/// narrow ones; several jobs split it, so a newly admitted one waits about
/// one round.
pub(super) const ROWS_PER_ROUND: usize = 1_600;
/// The least a turn advances a job by: 1.6 s, so a crowded round still
/// amortises its launches.
pub(super) const MIN_ROWS_PER_TURN: usize = 160;

/// One admitted job.
struct Active {
    events: mpsc::UnboundedSender<TranscriptEvent>,
    stream: Stream,
    mel: MelFrames,
    next_row: usize,
    detok: Detokenizer,
    pieces: Vec<TimedPiece>,
    /// The diarizer thread's answer, while its pass runs beside this job's
    /// recognition.
    diarizing: Option<std_mpsc::Receiver<diarize::Reply>>,
    activity: Option<SpeakerActivity>,
    /// The cuts a chunking stream was opened with, in seconds: its words
    /// take their chunk's speaker.
    cut_secs: Option<Vec<f64>>,
    segments_sent: usize,
    duration: f64,
    /// Whether the stream has been told the input is over.
    flushed: bool,
}

/// A job's recognizer stream, or what opening it waits for.
enum Stream {
    Open(Box<dyn RecognizerStream>),
    /// A chunking recognizer on a diarized job: its stream opens with the
    /// speaker turns as cuts, so it waits for the activity with the
    /// language it will be opened in.
    AwaitingTurns(Language),
}

/// What a turn left the job as.
enum Turn {
    More,
    Finished,
    Dropped,
}

/// Where a diarized job's pass is sent: the diarizer thread, when one runs.
pub(super) type DiarizerQueue = Option<std_mpsc::Sender<diarize::Request>>;

/// The thread body: runs until every service handle is gone.
pub(super) fn run(
    desc: &AudioServiceDesc,
    mut rx: mpsc::Receiver<Queued>,
    diarizer: &DiarizerQueue,
) {
    let mut active: Vec<Active> = Vec::new();
    loop {
        if active.is_empty() {
            let Some(queued) = rx.blocking_recv() else {
                return;
            };
            admit(desc, diarizer, queued, &mut active);
        }
        while active.len() < desc.max_streams {
            match rx.try_recv() {
                Ok(queued) => admit(desc, diarizer, queued, &mut active),
                Err(_) => break,
            }
        }
        let rows_per_turn = (ROWS_PER_ROUND / active.len().max(1)).max(MIN_ROWS_PER_TURN);
        let mut i = 0;
        while i < active.len() {
            // A panic in one job's model code ends that job, not the thread:
            // the others are mid-stream and did nothing wrong.
            let turn = catch_unwind(AssertUnwindSafe(|| {
                advance(desc, &mut active[i], rows_per_turn)
            }));
            match turn {
                Ok(Turn::More) => i += 1,
                Ok(Turn::Finished | Turn::Dropped) => {
                    active.swap_remove(i);
                }
                Err(_panic) => {
                    tracing::error!(
                        "a transcription job panicked; it is dropped and the worker goes on"
                    );
                    let _gone = active[i]
                        .events
                        .send(TranscriptEvent::Failed(internal_failure()));
                    active.swap_remove(i);
                }
            }
        }
    }
}

fn internal_failure() -> JobFailure {
    JobFailure {
        message: "the transcription failed inside this server".to_owned(),
        caller: false,
    }
}

/// Starts a job, or reports why it could not start.
fn admit(
    desc: &AudioServiceDesc,
    diarizer: &DiarizerQueue,
    queued: Queued,
    active: &mut Vec<Active>,
) {
    let Queued { job, events } = queued;
    // A caller that gave up while its job waited is not worth a mel pass.
    if events.is_closed() {
        return;
    }
    let started = catch_unwind(AssertUnwindSafe(|| {
        start(desc, diarizer, job, events.clone())
    }));
    let Ok(started) = started else {
        tracing::error!("starting a transcription job panicked; it is dropped");
        let _gone = events.send(TranscriptEvent::Failed(internal_failure()));
        return;
    };
    match started {
        Ok(job) => active.push(job),
        Err(error) => {
            // Before the first row runs, a refusal is about the request: a
            // language the model does not know, audio too short to frame.
            // A device error here is still this server's.
            let caller = !matches!(error, AudioError::Device(_) | AudioError::Io { .. });
            let _gone = events.send(TranscriptEvent::Failed(JobFailure {
                message: error.to_string(),
                caller,
            }));
        }
    }
}

fn mel_for(spec: &shiso_audio::MelSpec, samples: &[f32]) -> Result<MelFrames, AudioError> {
    let all = MelFrontend::new(spec.clone())?.compute(samples);
    let n = all.n_mels();
    let valid = all.valid();
    MelFrames::from_rows(all.as_slice()[..valid * n].to_vec(), n)
}

fn start(
    desc: &AudioServiceDesc,
    diarizer: &DiarizerQueue,
    job: TranscriptionJob,
    events: mpsc::UnboundedSender<TranscriptEvent>,
) -> Result<Active, AudioError> {
    let TranscriptionJob {
        pcm,
        language,
        diarize,
    } = job;
    let recognizer: &Arc<dyn SpeechRecognizer> = &desc.recognizer;
    let mel = mel_for(recognizer.mel_spec(), pcm.samples())?;
    if mel.valid() == 0 {
        return Err(AudioError::Empty);
    }
    // The whole clip is diarized in one pass, on the diarizer's thread while
    // the words are recognised here: speaker indices are only comparable
    // within one pass, and a segment cannot name its speaker until the
    // activity under it is known.
    let diarizing = match (diarize, &desc.diarizer, diarizer) {
        (false, _, _) => None,
        (true, Some(model), Some(queue)) => {
            let (reply, answer) = std_mpsc::channel();
            queue
                .send(diarize::Request {
                    mel: mel_for(model.mel_spec(), pcm.samples())?,
                    events: events.clone(),
                    reply,
                })
                .map_err(|_| AudioError::Io {
                    what: "handing the clip to the diarizer thread",
                    source: std::io::Error::other("the diarizer thread has stopped"),
                })?;
            Some(answer)
        }
        (true, _, _) => {
            return Err(AudioError::BadSpec {
                reason: "no diarizer is loaded".into(),
            });
        }
    };
    // A recognizer that chunks is better off cutting at the speaker turns
    // than at silence, so its stream waits for them; one that does not chunk
    // runs beside its diarizer as before.
    let stream = if diarizing.is_some() && recognizer.chunks_at_cuts() {
        Stream::AwaitingTurns(language)
    } else {
        Stream::Open(recognizer.open_stream(&StreamOptions {
            language,
            lookahead: desc.lookahead,
            cuts: None,
        })?)
    };
    Ok(Active {
        events,
        stream,
        mel,
        next_row: 0,
        detok: Detokenizer::new(Arc::clone(recognizer.vocab())),
        pieces: Vec::new(),
        diarizing,
        activity: None,
        cut_secs: None,
        segments_sent: 0,
        duration: pcm.duration_secs(),
        flushed: false,
    })
}

/// One turn of one job: the next rows, or, once they are all in, the flush;
/// then, once recognised, the transcript as soon as its speakers are known.
fn advance(desc: &AudioServiceDesc, job: &mut Active, rows_per_turn: usize) -> Turn {
    // A caller that has gone is noticed here, not only when a send fails: a
    // silent clip sends nothing, and would otherwise run to its end.
    if job.events.is_closed() {
        return Turn::Dropped;
    }
    if let Some(turn) = collect_speakers(desc, job, None) {
        return turn;
    }
    if let Stream::AwaitingTurns(_) = job.stream {
        // Waiting on the diarizer here, rather than spinning, is the same
        // bounded wait a recognised job takes for its speakers.
        if job.activity.is_none()
            && let Some(turn) = collect_speakers(desc, job, Some(SPEAKER_WAIT))
        {
            return turn;
        }
        if let Some(turn) = open_at_turns(desc, job) {
            return turn;
        }
    }
    if !job.flushed {
        let Stream::Open(stream) = &mut job.stream else {
            // Still waiting for the speaker turns.
            return Turn::More;
        };
        let total = job.mel.valid();
        let n = job.mel.n_mels();
        let result = if job.next_row < total {
            let end = (job.next_row + rows_per_turn).min(total);
            let rows = job.mel.as_slice()[job.next_row * n..end * n].to_vec();
            job.next_row = end;
            MelFrames::from_rows(rows, n).and_then(|chunk| stream.push(&chunk))
        } else {
            job.flushed = true;
            stream.finish()
        };
        match result {
            Err(error) => {
                let _gone = job.events.send(TranscriptEvent::Failed(JobFailure {
                    message: error.to_string(),
                    caller: false,
                }));
                return Turn::Finished;
            }
            Ok(tokens) if !deliver(desc, job, &tokens) => return Turn::Dropped,
            Ok(_) if !job.flushed => return Turn::More,
            Ok(_) => {}
        }
    }
    if job.diarizing.is_some() {
        if let Some(turn) = collect_speakers(desc, job, Some(SPEAKER_WAIT)) {
            return turn;
        }
        if job.diarizing.is_some() {
            return Turn::More;
        }
    }
    let _gone = job.events.send(TranscriptEvent::Done(finish(desc, job)));
    Turn::Finished
}

/// Opens a waiting stream once the job's speaker activity is known, cut at
/// its turns; `Some` when opening failed and that ends the job.
fn open_at_turns(desc: &AudioServiceDesc, job: &mut Active) -> Option<Turn> {
    let (Stream::AwaitingTurns(language), Some(activity)) = (&job.stream, &job.activity) else {
        return None;
    };
    let spec = desc.recognizer.mel_spec();
    let mut mel_frame_secs = 0.0;
    let opened = u32::try_from(spec.hop_length)
        .map_err(|_| AudioError::BadSpec {
            reason: format!("a hop of {} samples is not a mel frame", spec.hop_length),
        })
        .and_then(|hop| {
            mel_frame_secs = f64::from(hop) / f64::from(spec.sample_rate);
            chunks::turn_cuts(activity, &job.mel, mel_frame_secs)
                .into_iter()
                .map(|cut| {
                    u32::try_from(cut).map_err(|_| AudioError::BadSpec {
                        reason: format!(
                            "a cut at mel frame {cut} does not fit a stream's frame index"
                        ),
                    })
                })
                .collect::<Result<Vec<u32>, AudioError>>()
        })
        .and_then(|cuts| {
            let secs = cuts
                .iter()
                .map(|&cut| f64::from(cut) * mel_frame_secs)
                .collect();
            desc.recognizer
                .open_stream(&StreamOptions {
                    language: language.clone(),
                    lookahead: desc.lookahead,
                    cuts: Some(cuts),
                })
                .map(|stream| (stream, secs))
        });
    match opened {
        Ok((stream, secs)) => {
            job.stream = Stream::Open(stream);
            job.cut_secs = Some(secs);
            None
        }
        Err(error) => {
            // The same split as at admission: a refusal of the request is the
            // caller's, a device error this server's.
            let caller = !matches!(error, AudioError::Device(_) | AudioError::Io { .. });
            let _gone = job.events.send(TranscriptEvent::Failed(JobFailure {
                message: error.to_string(),
                caller,
            }));
            Some(Turn::Finished)
        }
    }
}

/// Takes the job's speaker activity if its pass has answered, waiting up to
/// `wait` for it, and sends the segments that settles; `Some` when that ends
/// the job.
fn collect_speakers(
    desc: &AudioServiceDesc,
    job: &mut Active,
    wait: Option<Duration>,
) -> Option<Turn> {
    let answer = job.diarizing.as_ref()?;
    let polled = match wait {
        None => answer
            .try_recv()
            .map_err(|e| e == std_mpsc::TryRecvError::Empty),
        Some(wait) => answer
            .recv_timeout(wait)
            .map_err(|e| e == std_mpsc::RecvTimeoutError::Timeout),
    };
    let failure = match polled {
        Err(true) => return None,
        Ok(Ok(activity)) => {
            job.activity = Some(activity);
            job.diarizing = None;
            // The words recognised so far are segmented the moment their
            // speakers are known, not at the next token.
            return (!deliver(desc, job, &[])).then_some(Turn::Dropped);
        }
        Ok(Err(error)) => JobFailure {
            message: error.to_string(),
            caller: false,
        },
        // Closed unanswered: the pass panicked.
        Err(false) => internal_failure(),
    };
    let _gone = job.events.send(TranscriptEvent::Failed(failure));
    Some(Turn::Finished)
}

/// Sends the text and any newly final segments; `false` when the caller has
/// gone.
fn deliver(desc: &AudioServiceDesc, job: &mut Active, tokens: &[TokenEvent]) -> bool {
    for event in tokens {
        let id = event.token.get();
        if desc.recognizer.is_special(id) {
            continue;
        }
        let Some(text) = job.detok.push(id) else {
            continue;
        };
        job.pieces.push(TimedPiece {
            text: text.clone(),
            frame: event.frame,
        });
        if job.events.send(TranscriptEvent::Delta(text)).is_err() {
            return false;
        }
    }
    if let Some(activity) = &job.activity {
        let words = words_of(desc, job);
        let speakers = speakers_of(job, &words, activity);
        let segments = transcript::diarized_segments(&words, &speakers);
        // Every segment but the last is final: a later piece can only extend
        // the last word, or start a word after it.
        let settled = segments.len().saturating_sub(1);
        for segment in segments.into_iter().take(settled).skip(job.segments_sent) {
            if job.events.send(TranscriptEvent::Segment(segment)).is_err() {
                return false;
            }
            job.segments_sent += 1;
        }
    }
    true
}

/// Each word's speaker: its chunk's, for a stream cut at the speaker turns,
/// since the recognizer's times inside a chunk are interpolated; otherwise
/// whoever is most active while it is spoken.
fn speakers_of(job: &Active, words: &[transcript::Word], activity: &SpeakerActivity) -> Vec<usize> {
    match &job.cut_secs {
        Some(cuts) => transcript::assign_speakers_by_chunk(words, activity, cuts),
        None => transcript::assign_speakers(words, activity),
    }
}

/// The job's words so far, each ending where its recognizer's timing says:
/// at the next word or its chunk's end for a chunking stream, one frame
/// after its last piece otherwise.
fn words_of(desc: &AudioServiceDesc, job: &Active) -> Vec<transcript::Word> {
    let chunk_ends = match &job.stream {
        Stream::Open(stream) => stream.chunk_ends(),
        Stream::AwaitingTurns(_) => &[],
    };
    transcript::words_in_chunks(&job.pieces, desc.recognizer.frame_secs(), chunk_ends)
}

fn finish(desc: &AudioServiceDesc, job: &Active) -> Transcript {
    let words = words_of(desc, job);
    let (segments, diarized) = match &job.activity {
        Some(activity) => {
            let speakers = speakers_of(job, &words, activity);
            (transcript::diarized_segments(&words, &speakers), true)
        }
        None => (transcript::segments(&words), false),
    };
    Transcript {
        // A word boundary the last piece opened is not text.
        text: job.detok.text().trim_end().to_owned(),
        // A stream that read the language out of the model's words knows it
        // better than the tags do; a stream that never opened knows nothing.
        language: match &job.stream {
            Stream::Open(stream) => stream.language(),
            Stream::AwaitingTurns(_) => None,
        }
        .or_else(|| job.detok.language())
        .map(str::to_owned),
        duration: job.duration,
        words,
        segments,
        diarized,
    }
}
