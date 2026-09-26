//! The arrival-order speaker cache and the offline chunk loop, on the host.
//!
//! The model scores at most eight speakers per step, and a speaker's index is
//! only its order of arrival within the context it was given. What makes an
//! index mean the same person across chunks is that every step also sees a
//! cache of earlier frames, chosen to be the most confidently single-speaker
//! frames of each speaker so far, grouped by speaker. This file keeps that
//! cache exactly as the reference does: which frames enter it, when it is
//! compressed, and how a compression scores and selects.
//!
//! Everything here is host arithmetic over at most a few thousand floats a
//! step, next to a device step over hundreds of rows of 512; so the cache
//! stays on the host and each step uploads its input whole.

use shiso_audio::{AudioError, MelFrames};

use super::config::OfflineParams;
use super::encoder::Encoder;

/// The speaker cache, the FIFO before it, and whether it has been compressed.
#[derive(Debug, Clone)]
pub struct SpeakerCache {
    params: OfflineParams,
    speakers: usize,
    width: usize,
    /// Pre-norm embeddings, `[cached][width]`.
    embeds: Vec<f32>,
    /// Their per-speaker probabilities, `[cached][speakers]`.
    probs: Vec<f32>,
    /// Pre-norm embeddings waiting to enter the cache, `[fifo][width]`.
    fifo: Vec<f32>,
    compressed: bool,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `floor(budget · rate)` for the small budgets a cache has.
fn share(budget: usize, rate: f32) -> usize {
    let budget = u16::try_from(budget).unwrap_or(u16::MAX);
    let value = (f32::from(budget) * rate).floor();
    // Non-negative and at most 1.5 x a u16 budget, so it fits.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a floored non-negative value below 2^17"
    )]
    let value = value as usize;
    value
}

impl SpeakerCache {
    /// An empty cache for `speakers` speakers over `width`-wide embeddings.
    #[must_use]
    pub const fn new(params: OfflineParams, speakers: usize, width: usize) -> Self {
        Self {
            params,
            speakers,
            width,
            embeds: Vec::new(),
            probs: Vec::new(),
            fifo: Vec::new(),
            compressed: false,
        }
    }

    fn cached(&self) -> usize {
        self.embeds.len() / self.width
    }

    fn fifo_len(&self) -> usize {
        self.fifo.len() / self.width
    }

    /// The step input: the cache, the FIFO, then `chunk` (its frames and the
    /// lookahead after them).
    #[must_use]
    pub fn inputs(&self, chunk: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.embeds.len() + self.fifo.len() + chunk.len());
        out.extend_from_slice(&self.embeds);
        out.extend_from_slice(&self.fifo);
        out.extend_from_slice(chunk);
        out
    }

    /// Folds one step into the cache: the step's `emitted` chunk frames enter
    /// the FIFO, and when it overflows its oldest frames move to the cache,
    /// which is compressed when it overflows in turn.
    ///
    /// `inputs` is the step input [`Self::inputs`] returned; `logits` are the
    /// step's `[rows·stack][speakers]` logits; `silence` is the learned
    /// silence embedding.
    pub fn update(
        &mut self,
        inputs: &[f32],
        logits: &[f32],
        stack: usize,
        silence: &[f32],
        emitted: usize,
    ) {
        let (w, s) = (self.width, self.speakers);
        let rows = inputs.len() / w;
        // Per-embedding probabilities: the stack's fine frames, averaged.
        let mut frame_probs = vec![0.0f32; rows * s];
        for f in 0..rows {
            for k in 0..s {
                let sum: f32 = (0..stack)
                    .map(|j| sigmoid(logits[(f * stack + j) * s + k]))
                    .sum();
                frame_probs[f * s + k] = sum / f32::from(u8::try_from(stack).unwrap_or(u8::MAX));
            }
        }
        let (nc, nf) = (self.cached(), self.fifo_len());
        let start = nc + nf;
        let mut fifo = std::mem::take(&mut self.fifo);
        fifo.extend_from_slice(&inputs[start * w..(start + emitted) * w]);
        let len = fifo.len() / w;
        if len > self.params.fifo {
            let pop = self
                .params
                .update_period
                .max(len - self.params.fifo)
                .min(len);
            // A compressed cache keeps the probabilities it was chosen with;
            // one never compressed is re-scored by this step.
            let mut probs = if self.compressed {
                std::mem::take(&mut self.probs)
            } else {
                frame_probs[..nc * s].to_vec()
            };
            probs.extend_from_slice(&frame_probs[nc * s..(nc + pop) * s]);
            let mut embeds = std::mem::take(&mut self.embeds);
            embeds.extend_from_slice(&fifo[..pop * w]);
            fifo.drain(..pop * w);
            if embeds.len() / w > self.params.cache {
                let n = embeds.len() / w;
                let kept = self.select(&probs, n);
                let mut new_embeds = Vec::with_capacity(kept.len() * w);
                let mut new_probs = Vec::with_capacity(kept.len() * s);
                for &f in &kept {
                    if f == n {
                        new_embeds.extend_from_slice(silence);
                        new_probs.extend(std::iter::repeat_n(0.0, s));
                    } else {
                        new_embeds.extend_from_slice(&embeds[f * w..(f + 1) * w]);
                        new_probs.extend_from_slice(&probs[f * s..(f + 1) * s]);
                    }
                }
                embeds = new_embeds;
                probs = new_probs;
                self.compressed = true;
            }
            self.embeds = embeds;
            self.probs = probs;
        }
        self.fifo = fifo;
    }

    /// The frames a compression keeps, in cache order, from `n` candidate
    /// frames with probabilities `probs` `[n][speakers]`. Index `n` stands
    /// for the silence embedding.
    ///
    /// Each speaker's frames are scored by how confidently they are that
    /// speaker alone; each speaker's best frames get a boost so every speaker
    /// keeps some; one silence frame per speaker always survives; then the
    /// best `cache` (speaker, frame) pairs are kept, grouped by speaker and in
    /// time order within one. Ties between equal scores break toward the lower
    /// index, which is where the reference leaves the order unspecified.
    #[must_use]
    pub fn select(&self, probs: &[f32], n: usize) -> Vec<usize> {
        let (s, p) = (self.speakers, &self.params);
        let budget = p.cache / s - p.silence_per_speaker;
        let min_pos = share(budget, p.min_pos_rate);
        let strong = share(budget, p.strong_rate);
        let weak = share(budget, p.weak_rate);
        let floor = p.score_threshold;
        let half = 0.5f32.ln();
        // scores[k][f]
        let mut scores = vec![vec![f32::NEG_INFINITY; n]; s];
        for f in 0..n {
            let row = &probs[f * s..(f + 1) * s];
            let not_sum: f32 = row.iter().map(|q| (1.0 - q).max(floor).ln()).sum();
            for k in 0..s {
                let q = row[k];
                if q > 0.5 {
                    scores[k][f] = q.max(floor).ln() - (1.0 - q).max(floor).ln() + not_sum - half;
                }
            }
        }
        for column in &mut scores {
            let positive = column.iter().filter(|v| **v > 0.0).count();
            if positive >= min_pos {
                // Enough confident frames: the overlapping ones go.
                for v in column.iter_mut() {
                    if *v <= 0.0 {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }
            for v in column.iter_mut().skip(p.cache) {
                *v += p.latest_boost;
            }
            for (count, boost) in [(strong, -2.0 * half), (weak, -half)] {
                for f in top(column, count) {
                    column[f] += boost;
                }
            }
        }
        // Speaker-major flat list, silence last in each speaker's block.
        let ns = n + 1;
        let mut flat: Vec<(f32, usize)> = Vec::with_capacity(s * ns);
        for (k, column) in scores.iter().enumerate() {
            flat.extend(column.iter().enumerate().map(|(f, v)| (*v, k * ns + f)));
            flat.push((f32::INFINITY, k * ns + n));
        }
        let sentinel = s * ns;
        let mut chosen: Vec<usize> = top_pairs(&flat, p.cache)
            .into_iter()
            .map(|(v, i)| if v == f32::NEG_INFINITY { sentinel } else { i })
            .collect();
        chosen.sort_unstable();
        chosen
            .into_iter()
            .map(|i| if i == sentinel { n } else { (i % ns).min(n) })
            .collect()
    }
}

/// Indices of the `count` largest values, ties toward the lower index.
fn top(values: &[f32], count: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*b].total_cmp(&values[*a]).then(a.cmp(b)));
    order.truncate(count);
    order
}

/// The `count` largest `(value, index)` pairs, ties toward the lower index.
fn top_pairs(pairs: &[(f32, usize)], count: usize) -> Vec<(f32, usize)> {
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    sorted.truncate(count);
    sorted
}

/// One clip's offline pass in progress: a step is one chunk, so a caller
/// serving several clips can interleave them.
#[derive(Debug)]
pub struct OfflinePass {
    cache: SpeakerCache,
    mel: MelFrames,
    frames: usize,
    start: usize,
    logits: Vec<f32>,
}

impl OfflinePass {
    /// Starts a pass over every valid row of `mel`.
    ///
    /// # Errors
    ///
    /// [`AudioError::Empty`] for no rows, [`AudioError::BadSpec`] for rows of
    /// the wrong width.
    pub fn new(encoder: &Encoder<'_>, mel: MelFrames) -> Result<Self, AudioError> {
        let c = encoder.config();
        let frames = encoder.frames(&mel)?;
        Ok(Self {
            cache: SpeakerCache::new(super::config::OFFLINE, c.speakers, c.d_model),
            mel,
            frames,
            start: 0,
            logits: Vec::with_capacity(frames * c.stack * c.speakers),
        })
    }

    /// Runs the next chunk: `None` while chunks remain, the probabilities
    /// `[mel rows][speakers]` after the last.
    ///
    /// # Errors
    ///
    /// A device error.
    pub fn step(
        &mut self,
        encoder: &Encoder<'_>,
        silence: &[f32],
    ) -> Result<Option<Vec<f32>>, AudioError> {
        let c = encoder.config();
        let params = super::config::OFFLINE;
        if self.start < self.frames {
            // Each step's window is projected when it runs, so a long clip
            // never holds every embedding at once; the lookahead frames a step
            // borrows are projected again as the next step's chunk, which
            // costs one small GEMM.
            let end = (self.start + params.chunk).min(self.frames);
            let emitted = end - self.start;
            let window_end = (end + params.lookahead).min(self.frames);
            let chunk = encoder.embed_window(&self.mel, self.start, window_end - self.start)?;
            let inputs = self.cache.inputs(&chunk);
            let before = self.cache.cached() + self.cache.fifo_len();
            let step = encoder.step(&inputs)?;
            let row = c.stack * c.speakers;
            self.logits
                .extend_from_slice(&step[before * row..(before + emitted) * row]);
            self.cache.update(&inputs, &step, c.stack, silence, emitted);
            self.start = end;
        }
        if self.start < self.frames {
            return Ok(None);
        }
        let mut logits = std::mem::take(&mut self.logits);
        logits.truncate(self.mel.valid() * c.speakers);
        Ok(Some(logits.into_iter().map(sigmoid).collect()))
    }
}

/// Speaker probabilities `[mel rows][speakers]` for one whole clip, as the
/// reference's offline mode computes them.
///
/// # Errors
///
/// [`AudioError::Empty`] for no rows, [`AudioError::BadSpec`] for rows of the
/// wrong width, or a device error.
pub fn run_offline(
    encoder: &Encoder<'_>,
    silence: &[f32],
    mel: &MelFrames,
) -> Result<Vec<f32>, AudioError> {
    let mut pass = OfflinePass::new(encoder, mel.clone())?;
    loop {
        if let Some(probs) = pass.step(encoder, silence)? {
            return Ok(probs);
        }
    }
}

/// `(speaker, start, end)` runs of `probs` `[frames][speakers]` above
/// `threshold`, ordered by start and then speaker.
#[must_use]
pub fn segments(probs: &[f32], speakers: usize, threshold: f32) -> Vec<(usize, usize, usize)> {
    let frames = probs.len() / speakers;
    let mut out = Vec::new();
    for k in 0..speakers {
        let mut open = None;
        for f in 0..=frames {
            let active = f < frames && probs[f * speakers + k] > threshold;
            match (active, open) {
                (true, None) => open = Some(f),
                (false, Some(a)) => {
                    out.push((k, a, f));
                    open = None;
                }
                _ => {}
            }
        }
    }
    out.sort_by_key(|&(k, a, _)| (a, k));
    out
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
