//! The audio tower: mel frames in, one `output_dim` row per audio token out,
//! the rows a prompt's audio placeholders are replaced by (notes §2).
//!
//! The clip is cut into 1-second chunks of [`CHUNK_FRAMES`] mel frames, and
//! each chunk goes through the conv stem on its own - no receptive field
//! crosses a chunk boundary. The tail chunk is zero-filled to a whole chunk
//! first and only its valid output tokens are kept: the stem's second and
//! third convs read `gelu(bias + ...)` in the filled region, not zeros, and
//! skipping the fill moves the tail's tokens by up to 1.19 (notes §2.1). Only
//! a clip shorter than one chunk runs unfilled, as the reference does.
//!
//! Every chunk's tokens then share one `[tokens][d_model]` matrix through the
//! transformer. Positions restart at 0 in every chunk (§2.3), and attention is
//! bidirectional inside blocks of [`WINDOW_TOKENS`] and blind across them
//! (§2.4) - the attention's chunk rule with no past chunks - so the whole clip
//! is one launch per operation rather than one per chunk or per window.
//!
//! Everything is f32; weights read in whatever width their export stored.

use shiso_audio::{AudioError, MelFrames};
use shiso_core::{DeviceBuffer, DeviceView};

use super::config::{AudioConfig, CHUNK_FRAMES, TOKENS_PER_CHUNK, WINDOW_TOKENS};
use super::weights::{Conv, Norm, TowerLayer, TowerWeights};
use crate::audio_ops::{
    AttnWindow, AxpyDesc, BiasRowsDesc, Conv2dGeometry, Im2colDesc, Linear, NormDesc, Ops,
    TransposeDesc, UnaryDesc, UnaryOp, WindowAttnDesc, Workspace,
};

/// Chunks one pass of the conv stem takes. The second conv's patches are
/// 14 MB per chunk, so a long clip goes through in groups of this many rather
/// than allocating over half a gigabyte for 40 seconds at once; at 8 the
/// launches per group are already few against the work in them.
const STEM_GROUP: usize = 8;

/// Output length of one stride-2, pad-1, 3-tap conv over `x` rows.
const fn halve(x: usize) -> usize {
    if x == 0 { 0 } else { (x - 1) / 2 + 1 }
}

/// Tokens the stem makes of `frames` mel frames in one chunk: three halvings.
const fn stem_tokens(frames: usize) -> usize {
    halve(halve(halve(frames)))
}

/// Audio tokens the tower makes of `frames` mel frames: 13 per whole chunk
/// and the stem's output length of the tail (notes §2.2). This sizes the
/// prompt's placeholder run before the tower runs.
#[must_use]
pub const fn audio_tokens(frames: usize) -> usize {
    (frames / CHUNK_FRAMES) * TOKENS_PER_CHUNK + stem_tokens(frames % CHUNK_FRAMES)
}

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

/// Rows `[row, row + count)` of a buffer of `width`-float rows.
fn rows(
    buf: &DeviceBuffer,
    row: usize,
    count: usize,
    width: usize,
) -> Result<DeviceView<'_>, AudioError> {
    Ok(buf.slice(bytes(row * width), bytes(count * width))?)
}

/// The tower's output for one clip.
#[derive(Debug)]
pub struct Encoded {
    /// `[tokens][output_dim]` f32.
    pub buffer: DeviceBuffer,
    /// Rows: [`audio_tokens`] of the clip's valid mel frames.
    pub tokens: usize,
}

/// The shape of one pass of the conv stem: how many chunks, how many frames
/// each, and every stage's extent.
#[derive(Debug, Clone, Copy)]
struct Stem {
    chunks: usize,
    /// `(time, frequency)` of the input and of each conv's output.
    extents: [(usize, usize); 4],
}

impl Stem {
    fn new(chunks: usize, frames: usize, mels: usize) -> Self {
        let mut extents = [(frames, mels); 4];
        for i in 1..4 {
            extents[i] = (halve(extents[i - 1].0), halve(extents[i - 1].1));
        }
        Self { chunks, extents }
    }

    fn tokens(&self) -> usize {
        self.extents[3].0
    }

    fn freq(&self) -> usize {
        self.extents[3].1
    }
}

/// Scratch shared by every layer, sized for one clip: slots of the
/// workspace.
struct Scratch<'w> {
    x: &'w DeviceBuffer,
    h: &'w DeviceBuffer,
    q: &'w DeviceBuffer,
    k: &'w DeviceBuffer,
    v: &'w DeviceBuffer,
    att: &'w DeviceBuffer,
    up: &'w DeviceBuffer,
}

/// The workspace slots of one encode: the layers' scratch, then one pass of
/// the stem's planes, patches and outputs, reused by every group.
mod slot {
    pub(super) const X: usize = 0;
    pub(super) const H: usize = 1;
    pub(super) const Q: usize = 2;
    pub(super) const K: usize = 3;
    pub(super) const V: usize = 4;
    pub(super) const ATT: usize = 5;
    pub(super) const UP: usize = 6;
    pub(super) const PLANES: usize = 7;
    /// Conv `i`'s patches.
    pub(super) const PATCHES: usize = 8;
    /// Conv `i`'s output.
    pub(super) const CONV: usize = 11;
    pub(super) const FLAT: usize = 14;
    pub(super) const COUNT: usize = 15;
}

/// The tower over one loaded checkpoint.
#[derive(Debug)]
pub struct Tower<'a> {
    ops: &'a Ops,
    weights: &'a TowerWeights,
    config: &'a AudioConfig,
}

impl<'a> Tower<'a> {
    /// Binds the tower to its weights.
    #[must_use]
    pub fn new(ops: &'a Ops, weights: &'a TowerWeights, config: &'a AudioConfig) -> Self {
        Self {
            ops,
            weights,
            config,
        }
    }

    /// Runs the tower over `mel`'s valid frames.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadSpec`] for mel rows of another width than the
    /// checkpoint's or a clip with no frames; a device error.
    pub fn encode(&self, mel: &MelFrames) -> Result<Encoded, AudioError> {
        self.encode_with(mel, &mut Workspace::new())
    }

    /// [`Self::encode`] in `ws`'s buffers, which a model keeps from one clip
    /// to the next.
    ///
    /// # Errors
    ///
    /// As [`Self::encode`].
    pub fn encode_with(&self, mel: &MelFrames, ws: &mut Workspace) -> Result<Encoded, AudioError> {
        let mels = self.config.mels;
        if mel.n_mels() != mels {
            return Err(AudioError::BadSpec {
                reason: format!("mel rows of {} bins for a tower of {mels}", mel.n_mels()),
            });
        }
        let frames = mel.valid();
        if frames == 0 {
            return Err(AudioError::BadSpec {
                reason: "a clip of no mel frames".into(),
            });
        }
        let tokens = audio_tokens(frames);
        // A clip shorter than a chunk is one unfilled chunk of its own length.
        let per_chunk = frames.min(CHUNK_FRAMES);
        let chunks = frames.div_ceil(per_chunk);
        let stem_rows = stem_tokens(per_chunk);
        ws.reserve(
            self.ops,
            &self.sizes(
                &Stem::new(STEM_GROUP.min(chunks), per_chunk, mels),
                chunks * stem_rows,
                tokens,
            ),
        )?;
        let ws = &*ws;
        let s = Scratch {
            x: ws.get(slot::X),
            h: ws.get(slot::H),
            q: ws.get(slot::Q),
            k: ws.get(slot::K),
            v: ws.get(slot::V),
            att: ws.get(slot::ATT),
            up: ws.get(slot::UP),
        };
        let valid = &mel.as_slice()[..frames * mels];
        let mut first = 0;
        while first < chunks {
            let group = STEM_GROUP.min(chunks - first);
            let stem = Stem::new(group, per_chunk, mels);
            self.stem(&stem, valid, first * per_chunk, ws, first * stem_rows)?;
            first += group;
        }
        // Only the tail chunk has rows past the clip's tokens, and it is last,
        // so the valid tokens are the matrix's first `tokens` rows.
        for layer in &self.weights.layers {
            self.layer(layer, &s, tokens)?;
        }
        self.head(&s, tokens)
    }

    /// Floats of every workspace slot, for a stem pass of `stem`'s shape, a
    /// hidden matrix of `rows` rows and `tokens` valid tokens.
    fn sizes(&self, stem: &Stem, rows: usize, tokens: usize) -> [usize; slot::COUNT] {
        let (mels, channels, d) = (
            self.config.mels,
            self.config.conv_channels,
            self.config.d_model,
        );
        let mut sizes = [0; slot::COUNT];
        sizes[slot::X] = rows * d;
        for i in [slot::H, slot::Q, slot::K, slot::V, slot::ATT] {
            sizes[i] = tokens * d;
        }
        sizes[slot::UP] = tokens * self.config.ffn;
        sizes[slot::PLANES] = stem.chunks * stem.extents[0].0 * mels;
        for i in 0..3 {
            let (t, f) = stem.extents[i + 1];
            let c_in = if i == 0 { 1 } else { channels };
            sizes[slot::PATCHES + i] = stem.chunks * t * f * 9 * c_in;
            sizes[slot::CONV + i] = stem.chunks * t * f * channels;
        }
        sizes[slot::FLAT] = stem.chunks * stem.tokens() * channels * stem.freq();
        sizes
    }

    /// The conv stem over `stem.chunks` chunks from mel frame `frame0`, whose
    /// `conv_out` rows, positions added, land at row `row0` of the hidden
    /// matrix.
    fn stem(
        &self,
        stem: &Stem,
        mel: &[f32],
        frame0: usize,
        ws: &Workspace,
        row0: usize,
    ) -> Result<(), AudioError> {
        let (mels, channels, width) = (
            self.config.mels,
            self.config.conv_channels,
            self.config.d_model,
        );
        let per_chunk = stem.extents[0].0;
        // The chunks as planes; past the clip's end, zeros (notes §2.1).
        let count = stem.chunks * per_chunk * mels;
        let mut planes = vec![0u8; count * 4];
        let end = (frame0 + stem.chunks * per_chunk).min(mel.len() / mels);
        for (to, v) in planes
            .chunks_exact_mut(4)
            .zip(&mel[frame0 * mels..end * mels])
        {
            to.copy_from_slice(&v.to_le_bytes());
        }
        let input = ws.get(slot::PLANES);
        self.ops
            .backend()
            .copy_h2d(input.slice(0, bytes(count))?, &planes)?;
        let mut current = input;
        let mut c_in = 1;
        for (i, conv) in self.weights.convs.iter().enumerate() {
            current = self.conv(stem, i, conv, current, c_in, ws)?;
            c_in = channels;
        }
        // `[chunk][t][f][c]` to `[chunk][t][c][f]`: the reference flattens a
        // token's stem output channel-major (notes §2.2).
        let (tokens, freq) = (stem.tokens(), stem.freq());
        let token_rows = stem.chunks * tokens;
        let flat = ws.get(slot::FLAT);
        self.ops.transpose(&TransposeDesc {
            out: flat.view(),
            input: current.view(),
            batch: token_rows,
            rows: freq,
            cols: channels,
        })?;
        let out = rows(ws.get(slot::X), row0, token_rows, width)?;
        self.weights
            .conv_out
            .forward(self.ops, flat.view(), out, token_rows)?;
        // Positions restart in every chunk: one `[tokens][width]` table per chunk.
        Ok(self.ops.bias_rows(&BiasRowsDesc {
            y: out,
            bias: rows(&self.weights.positions, 0, tokens, width)?,
            rows: stem.chunks,
            cols: tokens * width,
        })?)
    }

    /// Conv `i` of the stem and its GELU, over every chunk at once: the
    /// patches, then the GEMM against the kernel, which lands channels-last
    /// `[chunk][t][f][c]` as the next conv reads it.
    fn conv<'w>(
        &self,
        stem: &Stem,
        i: usize,
        conv: &Conv,
        input: &DeviceBuffer,
        c_in: usize,
        ws: &'w Workspace,
    ) -> Result<&'w DeviceBuffer, AudioError> {
        let c_out = self.config.conv_channels;
        let ((t_in, f_in), (t_out, f_out)) = (stem.extents[i], stem.extents[i + 1]);
        let positions = stem.chunks * t_out * f_out;
        let patches = ws.get(slot::PATCHES + i);
        self.ops.im2col_s2(&Im2colDesc {
            out: patches.view(),
            input: input.view(),
            geometry: Conv2dGeometry {
                t_in,
                f_in,
                c_in,
                t_out,
                f_out,
                c_out,
                pad_t: 1,
                pad_f: 1,
                depthwise: false,
                in_valid: t_in,
                out_valid: t_out,
                batch: stem.chunks,
            },
        })?;
        let out = ws.get(slot::CONV + i);
        self.linear(&conv.linear, patches, out, positions)?;
        self.gelu(out, positions * c_out)?;
        Ok(out)
    }

    fn norm(
        &self,
        norm: &Norm,
        x: &DeviceBuffer,
        out: &DeviceBuffer,
        n: usize,
    ) -> Result<(), AudioError> {
        Ok(self.ops.layer_norm(&NormDesc {
            out: out.view(),
            input: x.view(),
            weight: Some(norm.weight.view()),
            bias: Some(norm.bias.view()),
            rows: n,
            dim: self.config.d_model,
            eps: self.config.norm_eps,
        })?)
    }

    fn gelu(&self, buf: &DeviceBuffer, count: usize) -> Result<(), AudioError> {
        Ok(self.ops.unary(&UnaryDesc {
            op: UnaryOp::GeluErf,
            out: buf.view(),
            input: buf.view(),
            count,
        })?)
    }

    fn residual(
        &self,
        x: &DeviceBuffer,
        add: &DeviceBuffer,
        count: usize,
    ) -> Result<(), AudioError> {
        Ok(self.ops.axpy(&AxpyDesc {
            y: x.view(),
            x: add.view(),
            alpha: 1.0,
            count,
        })?)
    }

    fn linear(
        &self,
        linear: &Linear,
        x: &DeviceBuffer,
        y: &DeviceBuffer,
        n: usize,
    ) -> Result<(), AudioError> {
        Ok(linear.forward(self.ops, x.view(), y.view(), n)?)
    }

    /// One pre-norm layer, in place on `s.x` (notes §2.5).
    fn layer(&self, layer: &TowerLayer, s: &Scratch<'_>, n: usize) -> Result<(), AudioError> {
        let d = self.config.d_model;
        self.norm(&layer.attn_norm, s.x, s.h, n)?;
        self.linear(&layer.q, s.h, s.q, n)?;
        self.linear(&layer.k, s.h, s.k, n)?;
        self.linear(&layer.v, s.h, s.v, n)?;
        self.ops.window_attention(&WindowAttnDesc {
            out: s.att.view(),
            q: s.q.view(),
            k: s.k.view(),
            v: s.v.view(),
            n_q: n,
            n_k: n,
            heads: self.config.heads,
            head_dim: self.config.head_dim,
            // A query sees the keys of its own block of `WINDOW_TOKENS` and
            // nothing before or after it; the last block is the remainder.
            window: AttnWindow {
                q_abs0: 0,
                k_abs0: 0,
                k_end: n,
                chunk: WINDOW_TOKENS,
                left_chunks: 0,
            },
            bias_uv: None,
            q_dim_scale: None,
            rel: None,
            scale: attention_scale(self.config.head_dim),
            ac_scale: 1.0,
            softcap: None,
        })?;
        self.linear(&layer.o, s.att, s.h, n)?;
        self.residual(s.x, s.h, n * d)?;
        self.norm(&layer.ffn_norm, s.x, s.h, n)?;
        self.linear(&layer.fc1, s.h, s.up, n)?;
        self.gelu(s.up, n * self.config.ffn)?;
        self.linear(&layer.fc2, s.up, s.h, n)?;
        self.residual(s.x, s.h, n * d)
    }

    /// `proj2(gelu(proj1(ln_post(x))))` (notes §2.6).
    fn head(&self, s: &Scratch<'_>, n: usize) -> Result<Encoded, AudioError> {
        let w = self.weights;
        self.norm(&w.ln_post, s.x, s.h, n)?;
        self.linear(&w.proj1, s.h, s.q, n)?;
        self.gelu(s.q, n * self.config.d_model)?;
        let out = self.ops.scratch(n * self.config.output_dim)?;
        self.linear(&w.proj2, s.q, &out, n)?;
        Ok(Encoded {
            buffer: out,
            tokens: n,
        })
    }
}

#[cfg(test)]
#[path = "encoder_tests.rs"]
mod tests;
