//! The language prompt, the prediction network, the joint, and greedy search.
//!
//! **Prompt.** The prompt MLP reads `[encoder frame | one-hot language]`, so
//! its first layer is a linear over the frame plus one column per language.
//! That column is folded into the layer's bias once per stream.
//!
//! **Greedy search.** At frame `t` the joint is scored against the current
//! prediction state. A blank moves to the next frame and leaves the state
//! alone; any other token is emitted at `t`, advances the prediction network,
//! and is scored again on the same frame - at most `max_symbols` times before
//! the search must move on. Because a blank never changes the state, a run of
//! frames can be scored in one pass and every row after the first non-blank
//! thrown away: that is the same answer as scoring them one by one.
//!
//! **On the device.** The search runs there, a step at a time: the window of
//! frames, the joint, the row argmax, a one-thread decision, and the
//! prediction network advanced under the decision's emit flag, so a step that
//! found only blanks leaves it as it was. Nothing a step decides has to reach
//! the host before the next step reads it, so a run of steps is queued and
//! read back once: on long-dialogue, some 35 round trips instead of 488, each
//! of which cost about 0.2 ms of latency on an M1 Max before any arithmetic.

use shiso_audio::{AudioError, Language, TokenEvent};
use shiso_core::{DeviceBuffer, DeviceView};

use super::config::AsrConfig;
use super::encoder::Encoded;
use super::weights::AsrWeights;
use crate::audio_ops::load::upload_f32;
use crate::audio_ops::{
    ArgmaxDesc, AxpyDesc, BiasRowsDesc, EmbedRowDesc, GREEDY_STATE_WORDS, GreedyStepDesc,
    GreedyWindowDesc, KeepDesc, LstmDesc, Ops, UnaryDesc, UnaryOp,
};

/// Frames scored in one pass while looking for the next non-blank.
const SCAN: usize = 32;
/// Most steps queued before the host reads them back.
const MAX_BATCH: usize = 32;
/// What the step kernel writes for a step that emitted nothing.
const NO_TOKEN: u32 = u32::MAX;

fn bytes(floats: usize) -> u64 {
    u64::try_from(floats).unwrap_or(u64::MAX).saturating_mul(4)
}

fn words(host: &[u8]) -> Vec<u32> {
    host.chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn to_u32(n: usize, what: &str) -> Result<u32, AudioError> {
    u32::try_from(n).map_err(|_| AudioError::Checkpoint {
        what: format!("{what} past 2^32"),
    })
}

/// One stream's prediction network and the projected frames not yet searched.
#[derive(Debug)]
pub struct DecoderState {
    /// The absolute frame the search is on, and the symbols emitted on it.
    t: usize,
    sym: usize,
    parts: Parts,
}

#[derive(Debug)]
struct Parts {
    prompt_bias: DeviceBuffer,
    h: Vec<DeviceBuffer>,
    c: Vec<DeviceBuffer>,
    dec: DeviceBuffer,
    /// Projected frames from absolute frame `base` on, `[rows][joint]`.
    pending: DeviceBuffer,
    pending_rows: usize,
    base: usize,
    /// The search's state words on the device, while a feed runs.
    state: DeviceBuffer,
    work: Work,
}

/// A step's intermediates, allocated once per stream: steps run in order on
/// one queue, so each reuses what the previous one is done with.
#[derive(Debug)]
struct Work {
    joint: DeviceBuffer,
    logits: DeviceBuffer,
    ids: DeviceBuffer,
    x: [DeviceBuffer; 2],
    gates: DeviceBuffer,
    recurrent: DeviceBuffer,
    dec: DeviceBuffer,
}

impl DecoderState {
    /// A fresh stream conditioned on `language`, with the prediction network
    /// already run once on blank - the state the search starts from.
    ///
    /// # Errors
    ///
    /// [`AudioError::UnknownLanguage`], or a device error.
    pub fn new(transducer: &Transducer<'_>, language: &Language) -> Result<Self, AudioError> {
        let (ops, c) = (transducer.ops, transducer.config);
        let bias = transducer.prompt_bias(language)?;
        let zeros = |n: usize| ops.scratch(n);
        let mut parts = Parts {
            prompt_bias: bias,
            h: (0..c.pred_layers)
                .map(|_| zeros(c.pred_hidden))
                .collect::<Result<_, _>>()?,
            c: (0..c.pred_layers)
                .map(|_| zeros(c.pred_hidden))
                .collect::<Result<_, _>>()?,
            dec: zeros(c.joint)?,
            pending: zeros(c.joint)?,
            pending_rows: 0,
            base: 0,
            state: zeros(GREEDY_STATE_WORDS)?,
            work: Work {
                joint: zeros(SCAN * c.joint)?,
                logits: zeros(SCAN * c.vocab)?,
                ids: zeros(SCAN)?,
                x: [zeros(c.pred_hidden)?, zeros(c.pred_hidden)?],
                gates: zeros(4 * c.pred_hidden)?,
                recurrent: zeros(4 * c.pred_hidden)?,
                dec: zeros(c.joint)?,
            },
        };
        // The state the search starts from is the network run on blank, as
        // if a step had emitted it.
        transducer.write_state(&parts, [0, 0, 0, c.blank, 1])?;
        transducer.advance(&mut parts)?;
        Ok(Self {
            t: 0,
            sym: 0,
            parts,
        })
    }
}

/// The transducer heads over one loaded checkpoint.
#[derive(Debug)]
pub struct Transducer<'a> {
    ops: &'a Ops,
    weights: &'a AsrWeights,
    config: &'a AsrConfig,
}

impl<'a> Transducer<'a> {
    /// Binds the heads to their weights.
    #[must_use]
    pub fn new(ops: &'a Ops, weights: &'a AsrWeights, config: &'a AsrConfig) -> Self {
        Self {
            ops,
            weights,
            config,
        }
    }

    fn prompt_bias(&self, language: &Language) -> Result<DeviceBuffer, AudioError> {
        let id = self.config.prompt_id(language)?;
        let width = self.config.prompt_hidden;
        let column = &self.weights.prompt_cols[id * width..(id + 1) * width];
        let bias: Vec<f32> = self
            .weights
            .prompt_b1
            .iter()
            .zip(column)
            .map(|(b, c)| b + c)
            .collect();
        upload_f32(self.ops, &bias)
    }

    fn project_with(
        &self,
        prompt_bias: &DeviceBuffer,
        enc: DeviceView<'_>,
        frames: usize,
    ) -> Result<DeviceBuffer, AudioError> {
        let (ops, w, c) = (self.ops, self.weights, self.config);
        let hidden = ops.scratch(frames * c.prompt_hidden)?;
        w.prompt_w1.forward(ops, enc, hidden.view(), frames)?;
        ops.bias_rows(&BiasRowsDesc {
            y: hidden.view(),
            bias: prompt_bias.view(),
            rows: frames,
            cols: c.prompt_hidden,
        })?;
        ops.unary(&UnaryDesc {
            op: UnaryOp::Relu,
            out: hidden.view(),
            input: hidden.view(),
            count: frames * c.prompt_hidden,
        })?;
        let fused = ops.scratch(frames * c.d_model)?;
        w.prompt_w2
            .forward(ops, hidden.view(), fused.view(), frames)?;
        let projected = ops.scratch(frames * c.joint)?;
        w.enc_proj
            .forward(ops, fused.view(), projected.view(), frames)?;
        Ok(projected)
    }

    /// Prompt fusion and the encoder projection over `frames` encoder rows.
    ///
    /// # Errors
    ///
    /// [`AudioError::UnknownLanguage`], or a device error.
    pub fn project(
        &self,
        language: &Language,
        enc: DeviceView<'_>,
        frames: usize,
    ) -> Result<DeviceBuffer, AudioError> {
        let bias = self.prompt_bias(language)?;
        self.project_with(&bias, enc, frames)
    }

    /// Projects `enc`'s frames, searches as far as they reach, and returns the
    /// tokens found.
    ///
    /// # Errors
    ///
    /// A device error.
    pub fn feed(
        &self,
        state: &mut DecoderState,
        enc: &Encoded,
    ) -> Result<Vec<TokenEvent>, AudioError> {
        let (ops, j) = (self.ops, self.config.joint);
        if enc.frames > 0 {
            let projected =
                self.project_with(&state.parts.prompt_bias, enc.buffer.view(), enc.frames)?;
            let parts = &mut state.parts;
            let rows = parts.pending_rows + enc.frames;
            let grown = ops.scratch(rows * j)?;
            if parts.pending_rows > 0 {
                ops.backend().copy_d2d(
                    grown.slice(0, bytes(parts.pending_rows * j))?,
                    parts.pending.slice(0, bytes(parts.pending_rows * j))?,
                )?;
            }
            ops.backend().copy_d2d(
                grown.slice(bytes(parts.pending_rows * j), bytes(enc.frames * j))?,
                projected.slice(0, bytes(enc.frames * j))?,
            )?;
            parts.pending = grown;
            parts.pending_rows = rows;
        }
        let found = self.search(state)?;
        // Frames the search has passed are never scored again.
        let parts = &mut state.parts;
        let consumed = state.t.saturating_sub(parts.base).min(parts.pending_rows);
        if consumed > 0 {
            let keep = parts.pending_rows - consumed;
            let kept = ops.scratch(keep.max(1) * j)?;
            if keep > 0 {
                ops.backend().copy_d2d(
                    kept.slice(0, bytes(keep * j))?,
                    parts.pending.slice(bytes(consumed * j), bytes(keep * j))?,
                )?;
            }
            parts.pending = kept;
            parts.pending_rows = keep;
            parts.base += consumed;
        }
        Ok(found)
    }

    /// Greedy search over every pending frame, in batches of steps queued on
    /// the device and read back once each.
    fn search(&self, state: &mut DecoderState) -> Result<Vec<TokenEvent>, AudioError> {
        let ops = self.ops;
        let parts = &mut state.parts;
        let avail = parts.pending_rows;
        let mut t = state.t - parts.base;
        let mut found = Vec::new();
        if t >= avail {
            return Ok(found);
        }
        self.write_state(
            parts,
            [
                to_u32(t, "a frame")?,
                to_u32(state.sym, "a symbol count")?,
                0,
                0,
                0,
            ],
        )?;
        // A step emits a token or moves past frames, so a batch is sized by
        // how many steps the last one took per frame: about two frames a step
        // to start with, which is speech's rate at 80 ms a frame.
        let mut batch = avail.div_ceil(2).clamp(1, MAX_BATCH);
        loop {
            let out = ops.scratch(2 * batch)?;
            for slot in 0..batch {
                self.step(parts, avail, out.view(), slot)?;
            }
            let mut host = vec![0u8; 8 * batch];
            ops.backend()
                .copy_d2h(out.slice(0, bytes(2 * batch))?, &mut host)?;
            for pair in words(&host).chunks_exact(2) {
                if pair[0] != NO_TOKEN {
                    let frame = parts.base + usize::try_from(pair[1]).unwrap_or(usize::MAX);
                    found.push(TokenEvent {
                        token: shiso_core::TokenId::new(pair[0]),
                        frame: to_u32(frame, "a stream's frame")?,
                    });
                }
            }
            let mut host = vec![0u8; 4 * GREEDY_STATE_WORDS];
            ops.backend().copy_d2h(parts.state.view(), &mut host)?;
            let s = words(&host);
            let now = usize::try_from(s[0]).unwrap_or(usize::MAX);
            state.sym = usize::try_from(s[1]).unwrap_or(usize::MAX);
            let moved = now.saturating_sub(t).max(1);
            t = now;
            if s[2] != 0 || t >= avail {
                break;
            }
            batch = ((avail - t) * batch).div_ceil(moved).clamp(1, MAX_BATCH);
        }
        state.t = parts.base + t;
        Ok(found)
    }

    /// One step, queued: score the window, decide, and advance on the
    /// decision.
    fn step(
        &self,
        parts: &mut Parts,
        avail: usize,
        out: DeviceView<'_>,
        slot: usize,
    ) -> Result<(), AudioError> {
        let (ops, c) = (self.ops, self.config);
        let w = &parts.work;
        ops.greedy_window(&GreedyWindowDesc {
            out: w.joint.view(),
            pending: parts.pending.view(),
            dec: parts.dec.view(),
            state: parts.state.view(),
            rows: SCAN,
            cols: c.joint,
            avail,
        })?;
        self.weights
            .head
            .forward(ops, w.joint.view(), w.logits.view(), SCAN)?;
        // A tie keeps the lowest id, as the reference's argmax does.
        ops.argmax_rows(&ArgmaxDesc {
            out: w.ids.view(),
            input: w.logits.view(),
            rows: SCAN,
            cols: c.vocab,
        })?;
        ops.greedy_step(&GreedyStepDesc {
            state: parts.state.view(),
            out,
            ids: w.ids.view(),
            rows: SCAN,
            avail,
            blank: c.blank,
            max_symbols: c.max_symbols,
            slot,
        })?;
        self.advance(parts)
    }

    /// The prediction network on the state's last token, kept only if the
    /// last step emitted it.
    fn advance(&self, parts: &mut Parts) -> Result<(), AudioError> {
        let (ops, w, c) = (self.ops, self.weights, self.config);
        let h = c.pred_hidden;
        let work = &parts.work;
        let state = parts.state.view();
        ops.embed_row(&EmbedRowDesc {
            out: work.x[0].view(),
            table: w.embed.view(),
            state,
            width: h,
            vocab: c.vocab,
        })?;
        let mut input = 0;
        for (l, layer) in w.lstm.iter().enumerate() {
            let x = work.x[input].view();
            layer.w_ih.forward(ops, x, work.gates.view(), 1)?;
            layer
                .w_hh
                .forward(ops, parts.h[l].view(), work.recurrent.view(), 1)?;
            ops.axpy(&AxpyDesc {
                y: work.gates.view(),
                x: work.recurrent.view(),
                alpha: 1.0,
                count: 4 * h,
            })?;
            ops.lstm_cell(&LstmDesc {
                x_out: work.x[1 - input].view(),
                h: parts.h[l].view(),
                c: parts.c[l].view(),
                gates: work.gates.view(),
                state,
                hidden: h,
            })?;
            input = 1 - input;
        }
        w.dec_proj
            .forward(ops, work.x[input].view(), work.dec.view(), 1)?;
        ops.keep(&KeepDesc {
            dst: parts.dec.view(),
            src: work.dec.view(),
            state,
            count: c.joint,
        })?;
        Ok(())
    }

    /// Writes the search state's first five words: frame, symbols, done, last
    /// token, emitted.
    fn write_state(&self, parts: &Parts, first: [u32; 5]) -> Result<(), AudioError> {
        let mut host = [0u32; GREEDY_STATE_WORDS];
        host[..5].copy_from_slice(&first);
        let raw: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.ops.backend().copy_h2d(parts.state.view(), &raw)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "transducer_tests.rs"]
mod tests;
