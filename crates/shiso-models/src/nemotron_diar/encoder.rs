//! The stacked-feature projection, the transformer and the head, on the device.
//!
//! One step is one call: the speaker cache, the FIFO and the chunk go in as
//! pre-norm embeddings, and the head's logits for every row of the step come
//! out at the mel frame rate. Positions restart at zero every step and
//! attention is bidirectional over the whole step, as the reference runs it;
//! the kernel's window rule is made to see everything by putting every key in
//! one chunk.

use shiso_audio::{AudioError, MelFrames};
use shiso_core::{DeviceBuffer, DeviceView};

use super::config::DiarConfig;
use super::weights::{DiarWeights, Norm, UPSAMPLE_TAPS};
use crate::audio_ops::load::{download_f32, upload_f32};
use crate::audio_ops::{
    AttnWindow, AxpyDesc, NormDesc, Ops, RopeDesc, UnaryDesc, UnaryOp, UnfoldDesc, WindowAttnDesc,
};

/// Epsilon of every layer norm.
const EPS: f32 = 1e-5;
/// The rotary base.
const THETA: f32 = 10_000.0;

/// The device half of the diarizer, bound to one loaded checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct Encoder<'a> {
    ops: &'a Ops,
    c: &'a DiarConfig,
    w: &'a DiarWeights,
}

impl<'a> Encoder<'a> {
    /// Binds the ops to a checkpoint.
    #[must_use]
    pub const fn new(ops: &'a Ops, c: &'a DiarConfig, w: &'a DiarWeights) -> Self {
        Self { ops, c, w }
    }

    /// The checkpoint's numbers.
    #[must_use]
    pub const fn config(&self) -> &'a DiarConfig {
        self.c
    }

    fn scratch(&self, floats: usize) -> Result<DeviceBuffer, AudioError> {
        Ok(self.ops.scratch(floats.max(1))?)
    }

    fn norm(
        &self,
        out: &DeviceBuffer,
        input: &DeviceBuffer,
        norm: &Norm,
        rows: usize,
    ) -> Result<(), AudioError> {
        Ok(self.ops.layer_norm(&NormDesc {
            out: out.view(),
            input: input.view(),
            weight: Some(norm.weight.view()),
            bias: Some(norm.bias.view()),
            rows,
            dim: self.c.d_model,
            eps: EPS,
        })?)
    }

    fn add(&self, y: &DeviceBuffer, x: &DeviceBuffer, count: usize) -> Result<(), AudioError> {
        Ok(self.ops.axpy(&AxpyDesc {
            y: y.view(),
            x: x.view(),
            alpha: 1.0,
            count,
        })?)
    }

    fn unary(&self, op: UnaryOp, x: DeviceView<'_>, count: usize) -> Result<(), AudioError> {
        Ok(self.ops.unary(&UnaryDesc {
            op,
            out: x,
            input: x,
            count,
        })?)
    }

    /// Stacks every group of `stack` mel rows frame-major (zero-padding the
    /// last group) and projects it: the pre-norm embeddings `[frames][d_model]`
    /// the speaker cache stores.
    ///
    /// # Errors
    ///
    /// [`AudioError::Empty`] for no valid rows, [`AudioError::BadSpec`] for
    /// rows of the wrong width, or a device error.
    pub fn embed(&self, mel: &MelFrames) -> Result<Vec<f32>, AudioError> {
        let frames = self.frames(mel)?;
        self.embed_window(mel, 0, frames)
    }

    /// Encoder frames `mel` stacks to: every group of `stack` valid rows, the
    /// last one possibly partial.
    ///
    /// # Errors
    ///
    /// [`AudioError::Empty`] for no valid rows, [`AudioError::BadSpec`] for
    /// rows of the wrong width.
    pub fn frames(&self, mel: &MelFrames) -> Result<usize, AudioError> {
        if mel.n_mels() != self.c.mels {
            return Err(AudioError::BadSpec {
                reason: format!("{} mel bins, not {}", mel.n_mels(), self.c.mels),
            });
        }
        match mel.valid() {
            0 => Err(AudioError::Empty),
            valid => Ok(valid.div_ceil(self.c.stack)),
        }
    }

    /// [`Self::embed`] for encoder frames `first..first + count` only, so a
    /// long clip is projected a step at a time rather than held whole.
    ///
    /// # Errors
    ///
    /// As [`Self::frames`], or [`AudioError::BadSpec`] for a window past the
    /// clip's frames; a device error.
    pub fn embed_window(
        &self,
        mel: &MelFrames,
        first: usize,
        count: usize,
    ) -> Result<Vec<f32>, AudioError> {
        let c = self.c;
        let frames = self.frames(mel)?;
        if count == 0 || first + count > frames {
            return Err(AudioError::BadSpec {
                reason: format!("frames {first}..{} of {frames}", first + count),
            });
        }
        // Row-major rows are already frame-major within a group, so stacking
        // is the valid rows followed by the padding the last group needs.
        let row = c.stack * c.mels;
        let valid = mel.valid() * c.mels;
        let start = (first * row).min(valid);
        let end = ((first + count) * row).min(valid);
        let mut stacked = mel.as_slice()[start..end].to_vec();
        stacked.resize(count * row, 0.0);
        let x = upload_f32(self.ops, &stacked)?;
        let y = self.scratch(count * c.d_model)?;
        self.w.proj.forward(self.ops, x.view(), y.view(), count)?;
        download_f32(self.ops, y.view(), count * c.d_model)
    }

    /// The transformer over one step's embeddings `[rows][d_model]`: its final
    /// normed output, on the device.
    ///
    /// # Errors
    ///
    /// [`AudioError::Empty`] for no rows, [`AudioError::BadSpec`] for a
    /// ragged input, or a device error.
    #[expect(
        clippy::many_single_char_names,
        reason = "the tensors keep the names the model math uses: c, d, n, h, a, q, k, v, t, f"
    )]
    pub fn encode(&self, embeds: &[f32]) -> Result<DeviceBuffer, AudioError> {
        let c = self.c;
        let d = c.d_model;
        if embeds.is_empty() || !embeds.len().is_multiple_of(d) {
            return Err(if embeds.is_empty() {
                AudioError::Empty
            } else {
                AudioError::BadSpec {
                    reason: format!("{} values are not whole {d}-wide rows", embeds.len()),
                }
            });
        }
        let n = embeds.len() / d;
        let x = upload_f32(self.ops, embeds)?;
        let h = self.scratch(n * d)?;
        let a = self.scratch(n * d)?;
        let q = self.scratch(n * d)?;
        let k = self.scratch(n * d)?;
        let v = self.scratch(n * d)?;
        let t = self.scratch(n * d)?;
        let f = self.scratch(n * c.ff)?;
        self.norm(&h, &x, &self.w.in_norm, n)?;
        let window = AttnWindow {
            q_abs0: 0,
            k_abs0: 0,
            k_end: n,
            chunk: n,
            left_chunks: 0,
        };
        let head_dim = u16::try_from(c.head_dim).map_err(|_| AudioError::BadSpec {
            reason: format!("a {}-wide head", c.head_dim),
        })?;
        let scale = 1.0 / f32::from(head_dim).sqrt();
        for layer in &self.w.layers {
            self.norm(&a, &h, &layer.norm1, n)?;
            layer.q.forward(self.ops, a.view(), q.view(), n)?;
            layer.k.forward(self.ops, a.view(), k.view(), n)?;
            layer.v.forward(self.ops, a.view(), v.view(), n)?;
            for x in [&q, &k] {
                self.ops.rope_half(&RopeDesc {
                    x: x.view(),
                    rows: n,
                    heads: c.heads,
                    head_dim: c.head_dim,
                    pos0: 0,
                    theta: THETA,
                })?;
            }
            self.ops.window_attention(&WindowAttnDesc {
                out: a.view(),
                q: q.view(),
                k: k.view(),
                v: v.view(),
                n_q: n,
                n_k: n,
                heads: c.heads,
                head_dim: c.head_dim,
                window,
                bias_uv: None,
                q_dim_scale: None,
                rel: None,
                scale,
                ac_scale: 1.0,
                softcap: None,
            })?;
            layer.o.forward(self.ops, a.view(), t.view(), n)?;
            self.add(&h, &t, n * d)?;
            self.norm(&a, &h, &layer.norm2, n)?;
            layer.fc1.forward(self.ops, a.view(), f.view(), n)?;
            self.unary(UnaryOp::GeluErf, f.view(), n * c.ff)?;
            layer.fc2.forward(self.ops, f.view(), t.view(), n)?;
            self.add(&h, &t, n * d)?;
        }
        self.norm(&a, &h, &self.w.final_norm, n)?;
        Ok(a)
    }

    /// The head over `rows` encoder rows: speaker logits `[rows·stack][speakers]`
    /// at the mel frame rate, on the host.
    ///
    /// # Errors
    ///
    /// A device error.
    pub fn head(&self, hidden: &DeviceBuffer, rows: usize) -> Result<Vec<f32>, AudioError> {
        let c = self.c;
        let hh = c.head_hidden;
        let fine = rows * c.stack;
        let z = self.scratch(rows * hh)?;
        let u = self.scratch(rows * UPSAMPLE_TAPS * hh)?;
        let up = self.scratch(fine * hh)?;
        let g = self.scratch(fine * hh)?;
        let logits = self.scratch(fine * c.speakers)?;
        self.w
            .enc_proj
            .forward(self.ops, hidden.view(), z.view(), rows)?;
        self.ops.unfold(&UnfoldDesc {
            out: u.view(),
            input: z.view(),
            rows,
            channels: hh,
            taps: UPSAMPLE_TAPS,
            pad: UPSAMPLE_TAPS / 2,
        })?;
        // `[rows][stack·head]` is `[rows·stack][head]` in the same bytes: output
        // frame `stack·t + s` is channel block `s` of encoder row `t`.
        self.w
            .upsample
            .forward(self.ops, u.view(), up.view(), rows)?;
        self.unary(UnaryOp::Relu, up.view(), fine * hh)?;
        self.w.dense.forward(self.ops, up.view(), g.view(), fine)?;
        self.unary(UnaryOp::Relu, g.view(), fine * hh)?;
        self.w
            .out
            .forward(self.ops, g.view(), logits.view(), fine)?;
        download_f32(self.ops, logits.view(), fine * c.speakers)
    }

    /// One step: the transformer and the head over `embeds`.
    ///
    /// # Errors
    ///
    /// As [`Self::encode`] and [`Self::head`].
    pub fn step(&self, embeds: &[f32]) -> Result<Vec<f32>, AudioError> {
        let rows = embeds.len() / self.c.d_model;
        let hidden = self.encode(embeds)?;
        self.head(&hidden, rows)
    }
}

#[cfg(test)]
#[path = "encoder_tests.rs"]
pub(crate) mod tests;
