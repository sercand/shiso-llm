//! A whole clip planned up front, transcribed a batch of spans at a time.
//!
//! The worker plans the clip into spans ([`shiso_audio::chunks::plan_offline`])
//! and each [`OfflineJob::step`] takes the next `batch` of those to
//! transcribe: each is encoded through the tower on its own, then all of them
//! go through the decoder together ([`super::batch`]), whose ids equal the
//! single path's. A span's ids become its events exactly as a streamed chunk's
//! do ([`chunk_events`], the join with the span before included), so a clip
//! planned as the stream would cut it transcribes to the stream's events.
//!
//! Skipped spans (`transcribe: false`, a long silence the plan leaves out)
//! are never encoded and write nothing, and a join looks past them to the
//! last text a transcribed span wrote.

use std::ops::Range;
use std::sync::Arc;

use shiso_audio::chunks::Span;
use shiso_audio::{AudioError, MelFrames, OfflineBatch, OfflineJob, SpanResult};

use super::Model;
use super::encoder::{Encoded, Tower};
use super::stream::{Memory, chunk_events, chunk_mel, frame_word, warn_at_cap};

/// The indices of the next at most `batch` spans to transcribe, from index
/// `from` of `spans`, and where the next batch starts looking: past the last
/// one taken, or past every span when fewer than `batch` were left.
pub(crate) fn next_batch(spans: &[Span], from: usize, batch: usize) -> (Vec<usize>, usize) {
    let mut picked = Vec::with_capacity(batch);
    let mut at = from;
    while at < spans.len() && picked.len() < batch {
        if spans[at].transcribe {
            picked.push(at);
        }
        at += 1;
    }
    (picked, at)
}

/// Refuses a plan whose spans do not follow one another from frame 0 without
/// a gap, or hold an empty one: frames would be transcribed twice or never.
pub(crate) fn check_plan(spans: &[Span]) -> Result<(), AudioError> {
    let mut at = 0;
    for span in spans {
        if span.start != at || span.end <= span.start {
            return Err(AudioError::BadSpec {
                reason: format!(
                    "an offline plan's span {}..{} does not follow frame {at}",
                    span.start, span.end
                ),
            });
        }
        at = span.end;
    }
    Ok(())
}

/// One clip's offline job.
pub(crate) struct QwenOffline {
    model: Arc<Model>,
    /// The language the prompt forces, by the model's name for it.
    forced: Option<&'static str>,
    spans: Vec<Span>,
    /// The first span the next step looks at.
    next: usize,
    batch: usize,
    /// The last character the spans have written.
    last_char: Option<char>,
    /// Always `Some` until the job is dropped and gives it back; only its
    /// tower scratch is used, the batch keeping memory of its own.
    memory: Option<Memory>,
}

impl QwenOffline {
    /// A job over `spans` (checked by [`check_plan`]), `batch` spans a step.
    pub(crate) fn new(
        model: Arc<Model>,
        forced: Option<&'static str>,
        spans: Vec<Span>,
        batch: usize,
    ) -> Result<Self, AudioError> {
        if batch == 0 {
            return Err(AudioError::BadSpec {
                reason: "an offline job of zero spans a step never ends".into(),
            });
        }
        check_plan(&spans)?;
        let memory = Memory::take(&model);
        Ok(Self {
            model,
            forced,
            spans,
            next: 0,
            batch,
            last_char: None,
            memory: Some(memory),
        })
    }

    /// The span's rows through the tower.
    fn encode(&mut self, mel: &MelFrames, span: Range<usize>) -> Result<Encoded, AudioError> {
        let m = &*self.model;
        let n = m.config.audio.mels;
        let rows = mel.as_slice()[span.start * n..span.end * n].to_vec();
        let chunk = chunk_mel(rows, n)?;
        let memory = self.memory.as_mut().ok_or_else(|| AudioError::Checkpoint {
            what: "an offline job without its device memory".into(),
        })?;
        Tower::new(&m.ops, &m.weights.tower, &m.config.audio).encode_with(&chunk, &mut memory.tower)
    }
}

impl Drop for QwenOffline {
    fn drop(&mut self) {
        if let Some(memory) = self.memory.take() {
            memory.put_back(&self.model);
        }
    }
}

impl OfflineJob for QwenOffline {
    // None: every transcribed span has been returned.
    fn step(&mut self, mel: &MelFrames) -> Result<Option<OfflineBatch>, AudioError> {
        let (picked, after) = next_batch(&self.spans, self.next, self.batch);
        if picked.is_empty() {
            self.next = after;
            return Ok(None);
        }
        let n = self.model.config.audio.mels;
        let covered = self.spans.last().map_or(0, |s| s.end);
        if mel.n_mels() != n || mel.valid() < covered {
            return Err(AudioError::BadSpec {
                reason: format!(
                    "{} frames of {} mel bins for a plan of {covered} frames of {n}",
                    mel.valid(),
                    mel.n_mels()
                ),
            });
        }
        let mut encoded = Vec::with_capacity(picked.len());
        for &i in &picked {
            let span = self.spans[i];
            encoded.push(self.encode(mel, span.start..span.end)?);
        }
        let chunks: Vec<&Encoded> = encoded.iter().collect();
        let ids = self.model.transcribe_batch(&chunks, self.forced)?;
        let mut out = Vec::with_capacity(picked.len());
        for (&i, ids) in picked.iter().zip(&ids) {
            let span = self.spans[i];
            let frames = frame_word(span.start)?..frame_word(span.end)?;
            warn_at_cap(ids, &frames);
            let chunk = chunk_events(
                &self.model.vocab,
                ids,
                self.forced,
                frames.clone(),
                self.last_char,
            );
            if let Some(last) = chunk.last_char() {
                self.last_char = Some(last);
            }
            out.push(SpanResult {
                span,
                tokens: chunk.events,
                chunk_end: frames.end,
                language: chunk.language,
            });
        }
        self.next = after;
        Ok(Some(OfflineBatch { spans: out }))
    }
}

#[cfg(test)]
#[path = "offline_tests.rs"]
mod tests;
