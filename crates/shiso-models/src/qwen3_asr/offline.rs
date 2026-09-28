//! A whole clip planned up front, transcribed a batch of spans at a time.
//!
//! The worker plans the clip into spans ([`shiso_audio::chunks::plan_offline`])
//! and each [`OfflineJob::step`] transcribes the next `batch` of them: each is
//! encoded through the tower on its own, then all of them go through the
//! decoder together ([`super::batch`]), whose ids equal the single path's. A
//! span's ids become its events exactly as a streamed chunk's do
//! ([`chunk_events`], the join with the span before included), so a clip
//! planned as the stream would cut it transcribes to the stream's events.
//!
//! **Which spans share a batch.** A batch decodes until its longest span is
//! done, so a batch of one long span and seven short ones runs the long one's
//! passes mostly for one sequence. The spans are therefore batched longest
//! first ([`Schedule`]): like lengths share a batch, and a batch's passes are
//! close to its every span's. A span's ids do not depend on its batch, so
//! this moves no id; a step hands back the spans in plan order, as many as
//! are done from the first not yet handed back, and holds the rest until the
//! spans before them are. On a 57-minute meeting (328 spans, M1 Max,
//! `Q8_0`) plan-order batches ran 4487 decode passes, 3.1 tokens a pass, and
//! longest-first ones 2375, 5.9 a pass: 106 s of decoding became 76 s.
//!
//! **A span that echoes the keywords.** After a batch decodes with the
//! context, the spans whose text [`echoes`] the keywords ([`to_redecode`])
//! are decoded once more together, from the same tower rows, with no
//! context, and those ids replace theirs: at most one more pass a batch, and
//! never a third decode of a span.
//!
//! Skipped spans (`transcribe: false`, a long silence the plan leaves out)
//! are never encoded and write nothing, and a join looks past them to the
//! last text a transcribed span wrote.

use std::ops::Range;
use std::sync::Arc;

use shiso_audio::chunks::Span;
use shiso_audio::{AudioError, MelFrames, OfflineBatch, OfflineJob, PieceVocab, SpanResult};

use super::Model;
use super::batch::BatchTranscriber;
use super::encoder::{Encoded, Tower};
use super::prompt::Conditioning;
use super::stream::{chunk_events, chunk_mel, echoes, frame_word, warn_at_cap};

/// Which spans a job's steps transcribe, and which of them it can hand back.
#[derive(Debug)]
pub(crate) struct Schedule {
    /// The transcribed spans' indices, longest first, ties in plan order.
    order: Vec<usize>,
    /// The next place in `order` a batch starts at.
    cursor: usize,
    batch: usize,
    /// Whether each span is transcribed at all.
    transcribed: Vec<bool>,
    /// Each span's ids once its batch has run, until it is handed back.
    done: Vec<Option<Vec<u32>>>,
    /// The first span, in plan order, not yet handed back.
    released: usize,
}

impl Schedule {
    /// The schedule of `spans`, `batch` of them a step.
    pub(crate) fn new(spans: &[Span], batch: usize) -> Self {
        let mut order: Vec<usize> = (0..spans.len()).filter(|&i| spans[i].transcribe).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(spans[i].end - spans[i].start));
        Self {
            order,
            cursor: 0,
            batch,
            transcribed: spans.iter().map(|s| s.transcribe).collect(),
            done: vec![None; spans.len()],
            released: 0,
        }
    }

    /// The next batch's spans, in plan order; empty once every transcribed
    /// span has had its batch.
    pub(crate) fn next(&mut self) -> Vec<usize> {
        let end = (self.cursor + self.batch).min(self.order.len());
        let mut picked = self.order[self.cursor..end].to_vec();
        self.cursor = end;
        picked.sort_unstable();
        picked
    }

    /// Records span `i`'s ids.
    pub(crate) fn finish(&mut self, i: usize, ids: Vec<u32>) {
        self.done[i] = Some(ids);
    }

    /// The spans that can be handed back, in plan order with their ids: from
    /// the first not yet handed back, every transcribed span that is done,
    /// passing over skipped ones, up to the first that is not done.
    pub(crate) fn release(&mut self) -> Vec<(usize, Vec<u32>)> {
        let mut out = Vec::new();
        while self.released < self.done.len() {
            let i = self.released;
            if self.transcribed[i] {
                let Some(ids) = self.done[i].take() else {
                    break;
                };
                out.push((i, ids));
            }
            self.released += 1;
        }
        out
    }
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

/// Which of a batch's spans, decoded with the context to `ids` (in batch
/// order), echo `keywords` and are decoded again without it: their places
/// in `ids`, in order. `forced` is the language the prompt forced.
pub(crate) fn to_redecode(
    vocab: &Arc<PieceVocab>,
    ids: &[Vec<u32>],
    forced: Option<&str>,
    keywords: &[Vec<String>],
) -> Vec<usize> {
    ids.iter()
        .enumerate()
        .filter(|(_, ids)| echoes(vocab, ids, forced, keywords))
        .map(|(k, _)| k)
        .collect()
}

/// One clip's offline job.
pub(crate) struct QwenOffline {
    model: Arc<Model>,
    /// The language the prompt forces, by the model's name for it.
    forced: Option<&'static str>,
    /// The system turn's ids, shared by every span.
    context: Arc<[u32]>,
    /// The keywords `context` was made from, as word sequences: a span that
    /// echoes them is decoded again without it.
    keywords: Arc<[Vec<String>]>,
    /// How many spans were decoded again for echoing the keywords.
    redecoded: usize,
    spans: Vec<Span>,
    schedule: Schedule,
    /// The last character the spans handed back have written.
    last_char: Option<char>,
}

impl QwenOffline {
    /// A job over `spans` (checked by [`check_plan`]), `batch` spans a step.
    pub(crate) fn new(
        model: Arc<Model>,
        forced: Option<&'static str>,
        context: Arc<[u32]>,
        keywords: Arc<[Vec<String>]>,
        spans: Vec<Span>,
        batch: usize,
    ) -> Result<Self, AudioError> {
        if batch == 0 {
            return Err(AudioError::BadSpec {
                reason: "an offline job of zero spans a step never ends".into(),
            });
        }
        check_plan(&spans)?;
        Ok(Self {
            model,
            forced,
            context,
            keywords,
            redecoded: 0,
            schedule: Schedule::new(&spans, batch),
            spans,
            last_char: None,
        })
    }

    /// The ids of the spans `picked` of `mel`: through the tower together,
    /// each as it would be alone, then through the decoder together, in the
    /// model's pooled batch memory - so a job holds no device memory between
    /// its steps, and the stream memories stay for the streams. The spans
    /// that echo the keywords are decoded again without the context; the
    /// count of them comes back with the ids.
    fn transcribe(
        &self,
        mel: &MelFrames,
        picked: &[usize],
    ) -> Result<(Vec<Vec<u32>>, usize), AudioError> {
        let m = &*self.model;
        let n = m.config.audio.mels;
        let chunks = picked
            .iter()
            .map(|&i| {
                let Range { start, end } = self.spans[i].start..self.spans[i].end;
                chunk_mel(mel.as_slice()[start * n..end * n].to_vec(), n)
            })
            .collect::<Result<Vec<MelFrames>, AudioError>>()?;
        m.with_batch_memory(|memory| {
            let tower = Tower::new(&m.ops, &m.weights.tower, &m.config.audio);
            let started = std::time::Instant::now();
            let mels: Vec<&MelFrames> = chunks.iter().collect();
            let encoded = tower.encode_many(&mels, &mut memory.tower)?;
            // The encodes' tail would otherwise be timed as the prefill's.
            m.ops.backend().sync()?;
            tracing::debug!(
                spans = chunks.len(),
                frames = chunks.iter().map(MelFrames::valid).sum::<usize>(),
                encode_ms = started.elapsed().as_secs_f64() * 1e3,
                "encoded a batch"
            );
            let refs: Vec<&Encoded> = encoded.iter().collect();
            let decoder = BatchTranscriber::new(&m.ops, &m.weights.text, &m.config.text);
            let mut ids = decoder.transcribe_with(
                memory,
                &refs,
                &Conditioning {
                    forced: self.forced,
                    context: &self.context,
                },
            )?;
            let again = to_redecode(&m.vocab, &ids, self.forced, &self.keywords);
            if !again.is_empty() {
                let rows: Vec<&Encoded> = again.iter().map(|&k| refs[k]).collect();
                let plain =
                    decoder.transcribe_with(memory, &rows, &Conditioning::forced(self.forced))?;
                for (&k, plain) in again.iter().zip(plain) {
                    ids[k] = plain;
                }
            }
            Ok((ids, again.len()))
        })
    }
}

impl OfflineJob for QwenOffline {
    // None: every transcribed span has been handed back.
    fn step(&mut self, mel: &MelFrames) -> Result<Option<OfflineBatch>, AudioError> {
        let picked = self.schedule.next();
        if !picked.is_empty() {
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
            let (ids, redecoded) = self.transcribe(mel, &picked)?;
            self.redecoded += redecoded;
            for (&i, ids) in picked.iter().zip(ids) {
                self.schedule.finish(i, ids);
            }
        }
        let ready = self.schedule.release();
        if picked.is_empty() && ready.is_empty() {
            if !self.keywords.is_empty() {
                tracing::debug!(
                    spans = self.spans.iter().filter(|s| s.transcribe).count(),
                    redecoded = self.redecoded,
                    "an offline job's spans decoded again for echoing the keywords"
                );
            }
            return Ok(None);
        }
        let mut out = Vec::with_capacity(ready.len());
        for (i, ids) in ready {
            let span = self.spans[i];
            let frames = frame_word(span.start)?..frame_word(span.end)?;
            warn_at_cap(&ids, &frames);
            let chunk = chunk_events(
                &self.model.vocab,
                &ids,
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
        Ok(Some(OfflineBatch { spans: out }))
    }
}

#[cfg(test)]
#[path = "offline_tests.rs"]
mod tests;
