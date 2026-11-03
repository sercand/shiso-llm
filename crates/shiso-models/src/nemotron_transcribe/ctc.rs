//! The timestamp head: the speech branch's rows doubled in time and scored
//! against the timestamp tokenizer's pieces, for the alignment in
//! [`super::align`].
//!
//! Each 80 ms row is projected to two 40 ms rows, run through a two-layer
//! rotary transformer, and added back at a tenth of the transformer's
//! output; a classifier scores every piece and the blank. The head reads
//! the clip in windows of 500 rows with 50 rows of context on either side,
//! as the reference does, so a long clip's times do not depend on how far
//! attention reaches.
//!
//! The head runs while the clip is encoded, before the decoder has written
//! the words it will time, so it keeps what does not depend on them: each
//! frame's classifier input and the log-sum-exp of all its logits. Once the
//! words are known only their pieces' rows of the classifier are needed,
//! and those few dot products are cheap enough to do on the host.

#![expect(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    reason = "a log-sum-exp is summed in f64 and stored as the f32 the logits were; the names follow the model math (c, d, t, u, w)"
)]

use shiso_audio::AudioError;

use super::align::head_windows;
use super::config::PeeConfig;
use super::spm::PieceId;
use super::weights::CtcHead;
use crate::audio_ops::load::{download_f32, upload_f32};
use crate::audio_ops::store::{Stored, StoredType, TensorStore, dequant};
use crate::audio_ops::{AxpyDesc, Ops};

/// Encoder rows each head window keeps.
const WINDOW: usize = 500;
/// Encoder rows of context the head reads on each side of a window.
const CONTEXT: usize = 50;

/// The head's output for one clip that does not depend on the transcript.
#[derive(Debug, Clone, PartialEq)]
pub struct CtcEvidence {
    /// `[frames][dim]`: each timestamp frame's classifier input.
    rows: Vec<f32>,
    /// `[frames]`: the log-sum-exp of each frame's logits over every class.
    lse: Vec<f32>,
    dim: usize,
}

/// `v` rounded to the nearest bf16, ties to even: the alignment reads the
/// head's log-probabilities at that precision, as the reference's does.
fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return v;
    }
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) & 0xffff_0000)
}

/// Runs `f` over the rows of `out` (`row_len` values each) on every core.
fn par_rows(out: &mut [f32], row_len: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let rows = out.len() / row_len.max(1);
    if rows == 0 {
        return;
    }
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk, block) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            scope.spawn(move || {
                for (i, row) in block.chunks_mut(row_len).enumerate() {
                    f(chunk * per + i, row);
                }
            });
        }
    });
}

/// Each row's log-sum-exp, `[rows][width]` in.
fn log_sum_exp_rows(logits: &[f32], width: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; logits.len() / width];
    par_rows(&mut out, 1, |r, slot| {
        let row = &logits[r * width..(r + 1) * width];
        let peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = row.iter().map(|v| f64::from(v - peak).exp()).sum();
        slot[0] = (f64::from(peak) + sum.ln()) as f32;
    });
    out
}

impl CtcEvidence {
    /// Timestamp frames: the upsampling factor times the encoder rows.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.lse.len()
    }

    /// The log-probabilities of `classes` at every frame,
    /// `[frames][classes]`, rounded to bf16.
    #[must_use]
    pub fn log_probs(&self, classes: &ClassRows) -> Vec<f32> {
        let k = classes.bias.len();
        let d = self.dim;
        let mut out = vec![0.0f32; self.frames() * k];
        par_rows(&mut out, k, |t, row| {
            let h = &self.rows[t * d..(t + 1) * d];
            for (c, slot) in row.iter_mut().enumerate() {
                let w = &classes.weight[c * d..(c + 1) * d];
                let dot: f32 = w.iter().zip(h).map(|(a, b)| a * b).sum();
                *slot = bf16_round(dot + classes.bias[c] - self.lse[t]);
            }
        });
        out
    }
}

/// The classifier rows of some classes, on the host.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassRows {
    /// `[classes][dim]`.
    weight: Vec<f32>,
    /// `[classes]`.
    bias: Vec<f32>,
}

impl ClassRows {
    /// The rows of `classes` from the export in `store`, `dim` wide.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] for a missing or misshapen classifier, or a
    /// class past it.
    pub fn read(store: &TensorStore, classes: &[PieceId], dim: usize) -> Result<Self, AudioError> {
        let w = store.require("ctc.classifier.weight")?;
        let b = dequant(&store.require("ctc.classifier.bias")?)?;
        let bad = |what: String| AudioError::Checkpoint { what };
        let width = match w.ty {
            StoredType::F32 => 4,
            StoredType::F16 | StoredType::Bf16 => 2,
            _ => {
                return Err(bad(format!(
                    "ctc.classifier.weight is {:?}; the timestamp head reads f32, f16 or bf16",
                    w.ty
                )));
            }
        };
        let vocab = b.len();
        if w.shape != [vocab as u64, dim as u64] {
            return Err(bad(format!(
                "ctc.classifier.weight is {:?}, not [{vocab}, {dim}]",
                w.shape
            )));
        }
        let mut weight = Vec::with_capacity(classes.len() * dim);
        let mut bias = Vec::with_capacity(classes.len());
        for &PieceId(id) in classes {
            let id = id as usize;
            let row = w
                .data
                .get(id * dim * width..(id + 1) * dim * width)
                .filter(|_| id < vocab)
                .ok_or_else(|| bad(format!("class {id} is past the classifier")))?;
            weight.extend(dequant(&Stored {
                data: row,
                shape: vec![1, dim as u64],
                ..w.clone()
            })?);
            bias.push(b[id]);
        }
        Ok(Self { weight, bias })
    }
}

/// The head over the speech branch's rows `speech` (`[rows][d_model]`).
///
/// # Errors
///
/// [`AudioError::Empty`] for no rows, [`AudioError::Checkpoint`] for a head
/// narrower or wider than the speech branch, or a device error.
pub fn evidence(
    ops: &Ops,
    head: &CtcHead,
    c: &PeeConfig,
    speech: &[f32],
) -> Result<CtcEvidence, AudioError> {
    let d = c.ctc.d_model;
    if d != c.asr.d_model {
        return Err(AudioError::Checkpoint {
            what: format!(
                "a {d}-wide timestamp head over {}-wide speech rows",
                c.asr.d_model
            ),
        });
    }
    let (u, vocab) = (c.ctc_upsample, c.ctc_vocab);
    let t = speech.len() / d;
    if t == 0 {
        return Err(AudioError::Empty);
    }
    let mut rows = Vec::with_capacity(t * u * d);
    let mut lse = Vec::with_capacity(t * u);
    for w in head_windows(t, WINDOW, CONTEXT, CONTEXT) {
        let n = w.context_end - w.context_start;
        let x = upload_f32(ops, &speech[w.context_start * d..w.context_end * d])?;
        // `[n][u·d]` is `[u·n][d]` in the same bytes: each row's u
        // timestamp frames, in order.
        let up = ops.scratch(n * u * d)?;
        head.upsample.forward(ops, x.view(), up.view(), n)?;
        let y = head.transformer.forward(ops, &up, n * u)?;
        ops.axpy(&AxpyDesc {
            y: up.view(),
            x: y.view(),
            alpha: c.ctc_residual,
            count: n * u * d,
        })?;
        let logits = ops.scratch(n * u * vocab)?;
        head.classifier
            .forward(ops, up.view(), logits.view(), n * u)?;
        let first = u * (w.start - w.context_start);
        let count = u * (w.end - w.start);
        let bytes = |at: usize, len: usize| (at * 4) as u64..((at + len) * 4) as u64;
        let r = bytes(first * d, count * d);
        rows.extend(download_f32(
            ops,
            up.slice(r.start, r.end - r.start)?,
            count * d,
        )?);
        let r = bytes(first * vocab, count * vocab);
        let l = download_f32(ops, logits.slice(r.start, r.end - r.start)?, count * vocab)?;
        lse.extend(log_sum_exp_rows(&l, vocab));
    }
    Ok(CtcEvidence { rows, lse, dim: d })
}

#[cfg(test)]
#[path = "ctc_tests.rs"]
mod tests;
