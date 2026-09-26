//! The cache-aware conformer encoder, fed incrementally and exact.
//!
//! The reference's whole-utterance pass applies the same chunked attention
//! mask a stream does, so there is one encoder here and "offline" is a stream
//! fed everything before it is finished. Being exact under any push schedule
//! comes down to three invariants:
//!
//! - **Subsampling.** Each stride-2 stage emits output row `m` from input rows
//!   `2m-2 ..= 2m`, with zeros before the first. A stage keeps the input rows
//!   its next output still needs, and emits a row only once all three exist;
//!   finishing adds the one zero row the whole-utterance pass pads on the right.
//! - **Attention.** A query sees its own chunk of `R+1` frames, future frames
//!   included, and `left` frames of past chunks. So frames leave the encoder
//!   only as whole chunks - or as the last, partial one at finish - and each
//!   layer keeps the last `left` keys and values it projected.
//! - **Convolution.** The depthwise conv is causal; each layer keeps the last
//!   `kernel - 1` rows of its input.
//!
//! Everything is f32; weights read in whatever width their export stored.

use shiso_audio::AudioError;
use shiso_core::{DeviceBuffer, DeviceView};

use super::config::AsrConfig;
use super::weights::{AsrWeights, Block, Norm};
use crate::audio_ops::{
    AttnWindow, AxpyDesc, Conv2dDesc, Conv2dGeometry, DwConv1dDesc, GluDesc, Linear, NormDesc, Ops,
    RelPos, TransposeDesc, UnaryDesc, UnaryOp, WindowAttnDesc,
};

/// Epsilon of every layer norm in the stack.
const EPS: f32 = 1e-5;
/// Frequency padding ahead of each subsampling stage.
const PAD_F: usize = 2;

/// `1 / sqrt(head_dim)`, the scale on every attention logit.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the kernel takes the scale as f32"
)]
fn attention_scale(head_dim: usize) -> f32 {
    (1.0 / crate::audio_ops::f64_of(head_dim).sqrt()) as f32
}

fn bytes(floats: usize) -> u64 {
    u64::try_from(floats).unwrap_or(u64::MAX).saturating_mul(4)
}

/// Rows `[row, row + rows)` of a buffer of `width`-float rows.
fn rows(
    buf: &DeviceBuffer,
    row: usize,
    count: usize,
    width: usize,
) -> Result<DeviceView<'_>, AudioError> {
    Ok(buf.slice(bytes(row * width), bytes(count * width))?)
}

/// A subsampling stage's retained input rows.
#[derive(Debug)]
struct Stage {
    /// Up to two rows of the stage's input.
    pending: DeviceBuffer,
    held: usize,
    /// Floats per input row: frequency bins times channels.
    width: usize,
    /// Frequency bins of the input.
    freq: usize,
    /// Channels of the input.
    channels: usize,
}

/// One layer's carried context.
#[derive(Debug)]
struct LayerCache {
    k: DeviceBuffer,
    v: DeviceBuffer,
    held: usize,
    conv: DeviceBuffer,
}

/// Everything one stream carries between pushes.
#[derive(Debug)]
pub struct EncoderState {
    lookahead: usize,
    stages: [Stage; 3],
    layers: Vec<LayerCache>,
    /// Subsampled frames waiting for a whole chunk.
    ready: DeviceBuffer,
    ready_rows: usize,
    /// Encoder frames already emitted.
    seen: usize,
    finished: bool,
}

/// Encoder frames produced by one push.
#[derive(Debug)]
pub struct Encoded {
    /// `[frames][d_model]`.
    pub buffer: DeviceBuffer,
    /// Rows.
    pub frames: usize,
}

impl EncoderState {
    /// A fresh stream at `lookahead`.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadLookahead`] for a lookahead the checkpoint was not
    /// trained for; a device error.
    pub fn new(ops: &Ops, config: &AsrConfig, lookahead: usize) -> Result<Self, AudioError> {
        if !config.lookaheads.contains(&lookahead) {
            return Err(AudioError::BadLookahead {
                got: lookahead,
                allowed: config.lookaheads.clone(),
            });
        }
        let chunk = lookahead + 1;
        if !config.att_left.is_multiple_of(chunk) {
            return Err(AudioError::Checkpoint {
                what: format!(
                    "a left context of {} is not whole chunks of {chunk}",
                    config.att_left
                ),
            });
        }
        let f = |x: usize| x / 2 + 1;
        let (f1, f2) = (f(config.mels), f(f(config.mels)));
        let stage = |freq: usize, channels: usize| -> Result<Stage, AudioError> {
            // Two zero rows: the whole-utterance pass pads two frames ahead.
            Ok(Stage {
                pending: ops.scratch(2 * freq * channels)?,
                held: 2,
                width: freq * channels,
                freq,
                channels,
            })
        };
        let d = config.d_model;
        let layers = (0..config.layers)
            .map(|_| {
                Ok(LayerCache {
                    k: ops.scratch(config.att_left * d)?,
                    v: ops.scratch(config.att_left * d)?,
                    held: 0,
                    conv: ops.scratch((config.conv_kernel - 1) * d)?,
                })
            })
            .collect::<Result<Vec<_>, AudioError>>()?;
        Ok(Self {
            lookahead,
            stages: [
                stage(config.mels, 1)?,
                stage(f1, config.sub_channels)?,
                stage(f2, config.sub_channels)?,
            ],
            layers,
            ready: ops.scratch(chunk * d)?,
            ready_rows: 0,
            seen: 0,
            finished: false,
        })
    }

    /// Encoder frames emitted so far.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.seen
    }
}

/// The encoder over one loaded checkpoint.
#[derive(Debug)]
pub struct Encoder<'a> {
    ops: &'a Ops,
    weights: &'a AsrWeights,
    config: &'a AsrConfig,
}

impl<'a> Encoder<'a> {
    /// Binds the encoder to its weights.
    #[must_use]
    pub fn new(ops: &'a Ops, weights: &'a AsrWeights, config: &'a AsrConfig) -> Self {
        Self {
            ops,
            weights,
            config,
        }
    }

    fn copy(&self, dst: DeviceView<'_>, src: DeviceView<'_>) -> Result<(), AudioError> {
        if !src.is_empty() {
            self.ops.backend().copy_d2d(dst, src)?;
        }
        Ok(())
    }

    /// Runs one subsampling stage over its pending rows plus `new` rows, and
    /// returns its output rows `[t_out][f_out][channels]` and their count.
    fn stage(
        &self,
        index: usize,
        stage: &mut Stage,
        new: Option<DeviceView<'_>>,
        new_rows: usize,
        finishing: bool,
    ) -> Result<(DeviceBuffer, usize), AudioError> {
        let c = self.config.sub_channels;
        let t_in = stage.held + new_rows;
        let t_out = if finishing {
            t_in / 2
        } else {
            t_in.saturating_sub(1) / 2
        };
        let input = self.ops.scratch(t_in * stage.width)?;
        self.copy(
            rows(&input, 0, stage.held, stage.width)?,
            rows(&stage.pending, 0, stage.held, stage.width)?,
        )?;
        if let Some(new) = new {
            self.copy(rows(&input, stage.held, new_rows, stage.width)?, new)?;
        }
        let f_out = stage.freq / 2 + 1;
        let out = self.ops.scratch((t_out * f_out * c).max(1))?;
        if t_out > 0 {
            let geometry = Conv2dGeometry {
                t_in,
                f_in: stage.freq,
                c_in: stage.channels,
                t_out,
                f_out,
                c_out: c,
                pad_t: 0,
                pad_f: PAD_F,
                depthwise: index > 0,
                in_valid: t_in,
                out_valid: t_out,
            };
            let (weight, bias) = match index {
                0 => (&self.weights.conv0, &self.weights.conv0_bias),
                s => (
                    &self.weights.stages[s - 1].dw,
                    &self.weights.stages[s - 1].dw_bias,
                ),
            };
            self.ops.conv2d_s2(&Conv2dDesc {
                out: out.view(),
                input: input.view(),
                weight: weight.view(),
                bias: Some(bias.view()),
                geometry,
                relu: index == 0,
            })?;
            if index > 0 {
                let pw = &self.weights.stages[index - 1].pw;
                let mixed = self.ops.scratch(t_out * f_out * c)?;
                pw.forward(self.ops, out.view(), mixed.view(), t_out * f_out)?;
                self.ops.unary(&UnaryDesc {
                    op: UnaryOp::Relu,
                    out: mixed.view(),
                    input: mixed.view(),
                    count: t_out * f_out * c,
                })?;
                self.keep(stage, &input, t_in, t_out)?;
                return Ok((mixed, t_out));
            }
        }
        self.keep(stage, &input, t_in, t_out)?;
        Ok((out, t_out))
    }

    /// Keeps the input rows the stage's next output still reads.
    fn keep(
        &self,
        stage: &mut Stage,
        input: &DeviceBuffer,
        t_in: usize,
        t_out: usize,
    ) -> Result<(), AudioError> {
        let from = 2 * t_out;
        let held = t_in - from;
        self.copy(
            rows(&stage.pending, 0, held, stage.width)?,
            rows(input, from, held, stage.width)?,
        )?;
        stage.held = held;
        Ok(())
    }

    fn norm(
        &self,
        norm: &Norm,
        x: DeviceView<'_>,
        out: DeviceView<'_>,
        n: usize,
    ) -> Result<(), AudioError> {
        Ok(self.ops.layer_norm(&NormDesc {
            out,
            input: x,
            weight: Some(norm.weight.view()),
            bias: Some(norm.bias.view()),
            rows: n,
            dim: self.config.d_model,
            eps: EPS,
        })?)
    }

    /// `x += alpha * down(silu(up(norm(x))))`.
    fn feed_forward(
        &self,
        norm: &Norm,
        ff: &(Linear, Linear),
        x: &DeviceBuffer,
        n: usize,
    ) -> Result<(), AudioError> {
        let d = self.config.d_model;
        let h = self.ops.scratch(n * d)?;
        self.norm(norm, x.view(), h.view(), n)?;
        let up = self.ops.scratch(n * self.config.ff)?;
        ff.0.forward(self.ops, h.view(), up.view(), n)?;
        self.ops.unary(&UnaryDesc {
            op: UnaryOp::Silu,
            out: up.view(),
            input: up.view(),
            count: n * self.config.ff,
        })?;
        ff.1.forward(self.ops, up.view(), h.view(), n)?;
        Ok(self.ops.axpy(&AxpyDesc {
            y: x.view(),
            x: h.view(),
            alpha: 0.5,
            count: n * d,
        })?)
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "q, k, v, h and x are the attention's own names"
    )]
    fn attention(
        &self,
        block: &Block,
        cache: &mut LayerCache,
        x: &DeviceBuffer,
        n: usize,
        seen: usize,
        lookahead: usize,
    ) -> Result<(), AudioError> {
        let d = self.config.d_model;
        let h = self.ops.scratch(n * d)?;
        self.norm(&block.norm_att, x.view(), h.view(), n)?;
        let total = cache.held + n;
        let (q, k, v) = (
            self.ops.scratch(n * d)?,
            self.ops.scratch(total * d)?,
            self.ops.scratch(total * d)?,
        );
        block.q.forward(self.ops, h.view(), q.view(), n)?;
        self.copy(
            rows(&k, 0, cache.held, d)?,
            rows(&cache.k, 0, cache.held, d)?,
        )?;
        self.copy(
            rows(&v, 0, cache.held, d)?,
            rows(&cache.v, 0, cache.held, d)?,
        )?;
        block
            .k
            .forward(self.ops, h.view(), rows(&k, cache.held, n, d)?, n)?;
        block
            .v
            .forward(self.ops, h.view(), rows(&v, cache.held, n, d)?, n)?;
        let chunk = lookahead + 1;
        let att = self.ops.scratch(n * d)?;
        self.ops.window_attention(&WindowAttnDesc {
            out: att.view(),
            q: q.view(),
            k: k.view(),
            v: v.view(),
            n_q: n,
            n_k: total,
            heads: self.config.heads,
            head_dim: self.config.head_dim,
            window: AttnWindow {
                q_abs0: seen,
                k_abs0: seen - cache.held,
                k_end: seen + n,
                chunk,
                left_chunks: self.config.att_left / chunk,
            },
            bias_uv: Some((block.bias_u.view(), block.bias_v.view())),
            q_dim_scale: None,
            rel: Some(RelPos {
                table: block.rel.view(),
                d_min: self.weights.rel_d_min,
                rows: self.weights.rel_rows,
            }),
            scale: attention_scale(self.config.head_dim),
            ac_scale: 1.0,
            softcap: None,
        })?;
        block.o.forward(self.ops, att.view(), h.view(), n)?;
        self.ops.axpy(&AxpyDesc {
            y: x.view(),
            x: h.view(),
            alpha: 1.0,
            count: n * d,
        })?;
        let keep = total.min(self.config.att_left);
        self.copy(
            rows(&cache.k, 0, keep, d)?,
            rows(&k, total - keep, keep, d)?,
        )?;
        self.copy(
            rows(&cache.v, 0, keep, d)?,
            rows(&v, total - keep, keep, d)?,
        )?;
        cache.held = keep;
        Ok(())
    }

    fn convolution(
        &self,
        block: &Block,
        cache: &LayerCache,
        x: &DeviceBuffer,
        n: usize,
    ) -> Result<(), AudioError> {
        let d = self.config.d_model;
        let taps = self.config.conv_kernel;
        let h = self.ops.scratch(n * d)?;
        self.norm(&block.norm_conv, x.view(), h.view(), n)?;
        let wide = self.ops.scratch(n * 2 * d)?;
        block.pw1.forward(self.ops, h.view(), wide.view(), n)?;
        let padded = self.ops.scratch((taps - 1 + n) * d)?;
        self.copy(
            rows(&padded, 0, taps - 1, d)?,
            rows(&cache.conv, 0, taps - 1, d)?,
        )?;
        self.ops.glu(&GluDesc {
            out: rows(&padded, taps - 1, n, d)?,
            input: wide.view(),
            rows: n,
            channels: d,
        })?;
        self.ops.dwconv1d(&DwConv1dDesc {
            out: h.view(),
            input: padded.view(),
            weight: block.dw.view(),
            rows: n,
            channels: d,
            taps,
        })?;
        self.norm(&block.conv_norm, h.view(), h.view(), n)?;
        self.ops.unary(&UnaryDesc {
            op: UnaryOp::Silu,
            out: h.view(),
            input: h.view(),
            count: n * d,
        })?;
        let out = self.ops.scratch(n * d)?;
        block.pw2.forward(self.ops, h.view(), out.view(), n)?;
        self.ops.axpy(&AxpyDesc {
            y: x.view(),
            x: out.view(),
            alpha: 1.0,
            count: n * d,
        })?;
        self.copy(
            rows(&cache.conv, 0, taps - 1, d)?,
            rows(&padded, n, taps - 1, d)?,
        )?;
        Ok(())
    }

    /// Runs every block over `n` frames in place.
    fn blocks(
        &self,
        state: &mut EncoderState,
        x: &DeviceBuffer,
        n: usize,
    ) -> Result<(), AudioError> {
        let (seen, lookahead) = (state.seen, state.lookahead);
        for (block, cache) in self.weights.layers.iter().zip(state.layers.iter_mut()) {
            self.feed_forward(&block.norm_ff1, &block.ff1, x, n)?;
            self.attention(block, cache, x, n, seen, lookahead)?;
            self.convolution(block, cache, x, n)?;
            self.feed_forward(&block.norm_ff2, &block.ff2, x, n)?;
            self.norm(&block.norm_out, x.view(), x.view(), n)?;
        }
        state.seen += n;
        Ok(())
    }

    fn run(
        &self,
        state: &mut EncoderState,
        mel: Option<DeviceView<'_>>,
        n_mel: usize,
        finishing: bool,
    ) -> Result<Encoded, AudioError> {
        if state.finished {
            return Err(AudioError::Checkpoint {
                what: "the encoder stream was already finished".into(),
            });
        }
        let d = self.config.d_model;
        // Subsampling: three stages, each consuming the last one's rows.
        let (mut cur, mut cur_rows) = self.stage(0, &mut state.stages[0], mel, n_mel, finishing)?;
        for s in 1..3 {
            let view = if cur_rows > 0 {
                Some(rows(&cur, 0, cur_rows, state.stages[s].width)?)
            } else {
                None
            };
            let (next, next_rows) =
                self.stage(s, &mut state.stages[s], view, cur_rows, finishing)?;
            cur = next;
            cur_rows = next_rows;
        }
        // Flatten channel-major: [t][f][c] -> [t][c][f], then project.
        let (f3, c) = (state.stages[2].freq / 2 + 1, self.config.sub_channels);
        let all = state.ready_rows + cur_rows;
        let frames = self.ops.scratch(all.max(1) * d)?;
        self.copy(
            rows(&frames, 0, state.ready_rows, d)?,
            rows(&state.ready, 0, state.ready_rows, d)?,
        )?;
        if cur_rows > 0 {
            let flat = self.ops.scratch(cur_rows * f3 * c)?;
            self.ops.transpose(&TransposeDesc {
                out: flat.view(),
                input: cur.view(),
                batch: cur_rows,
                rows: f3,
                cols: c,
            })?;
            self.weights.sub_out.forward(
                self.ops,
                flat.view(),
                rows(&frames, state.ready_rows, cur_rows, d)?,
                cur_rows,
            )?;
        }
        // Whole chunks leave; at finish the partial one does too. Every chunk
        // this push completed goes through the blocks in one pass: attention
        // applies the chunk rule itself, so a query sees exactly the keys a
        // chunk-at-a-time pass would have shown it, and the conv carries its
        // causal state across the pass. One pass instead of one per chunk is
        // a few wide launches instead of many narrow ones.
        let chunk = state.lookahead + 1;
        let emit = if finishing { all } else { all / chunk * chunk };
        if emit > 0 {
            let x = self.ops.scratch(emit * d)?;
            self.copy(x.view(), rows(&frames, 0, emit, d)?)?;
            self.blocks(state, &x, emit)?;
            self.copy(rows(&frames, 0, emit, d)?, x.view())?;
        }
        state.ready_rows = all - emit;
        self.copy(
            rows(&state.ready, 0, state.ready_rows, d)?,
            rows(&frames, emit, state.ready_rows, d)?,
        )?;
        if finishing {
            state.finished = true;
        }
        let out = self.ops.scratch(emit.max(1) * d)?;
        self.copy(rows(&out, 0, emit, d)?, rows(&frames, 0, emit, d)?)?;
        Ok(Encoded {
            buffer: out,
            frames: emit,
        })
    }

    /// Adds `n_mel` mel rows and returns the encoder frames that became final.
    ///
    /// # Errors
    ///
    /// A device error, or [`AudioError::Checkpoint`] after the stream finished.
    pub fn push(
        &self,
        state: &mut EncoderState,
        mel: DeviceView<'_>,
        n_mel: usize,
        finishing: bool,
    ) -> Result<Encoded, AudioError> {
        self.run(state, Some(mel), n_mel, finishing)
    }

    /// Flushes every frame the whole-utterance pass would still produce.
    ///
    /// # Errors
    ///
    /// A device error, or [`AudioError::Checkpoint`] after the stream finished.
    pub fn finish(&self, state: &mut EncoderState) -> Result<Encoded, AudioError> {
        self.run(state, None, 0, true)
    }
}

#[cfg(test)]
#[path = "encoder_tests.rs"]
mod tests;
