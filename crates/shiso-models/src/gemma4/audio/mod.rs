//! The Gemma 4 audio tower: a conformer over log-mel frames, projected into the
//! language model's embedding space.
//!
//! The graph, in order:
//!
//! 1. Two stride-2 `3x3` convolutions over time and frequency, each followed by a
//!    layer norm over channels (weight only) and a rectifier; flattened
//!    frequency-major and projected to the model width.
//! 2. Conformer layers: a half-step feed-forward, local self-attention, a
//!    light convolution, a second half-step feed-forward, a closing norm.
//!    Every norm is an RMS norm; most linears clamp their input and output to
//!    bounds the checkpoint records. Attention is chunked-local with twelve
//!    keys per query - the query and the eleven before it - a relative-position
//!    term, a per-dimension query scale and a tanh logit cap.
//! 3. An output projection, a scale-free RMS norm and the projection into the
//!    language model's width.
//!
//! One clip is encoded offline: the whole of its mel rows at once, no caches.

mod loader;

use std::path::Path;
use std::sync::Arc;

use shiso_audio::{
    AudioClip, AudioError, AudioTower, Filterbank, Framing, LogScale, MelScale, MelSpec,
    Preemphasis, Spectrum, Window,
};
use shiso_core::{Backend, DeviceBuffer, DeviceView, VisionGrid};
use shiso_vision::EncodedRows;

use crate::audio_ops::load::upload_f32;
use crate::audio_ops::{
    AttnWindow, AxpyDesc, Conv2dDesc, Conv2dGeometry, DwConv1dDesc, GluDesc, NormDesc, Ops, RelPos,
    UnaryDesc, UnaryOp, WindowAttnDesc,
};
use loader::{AudioWeights, Clipped, FeedForward, Layer};

/// Soft tokens the tower produces at most: thirty seconds at 40 ms a token.
const MAX_TOKENS: usize = 750;
/// Epsilon of every norm in the tower.
const EPS: f32 = 1e-6;
/// The logit cap of the attention.
const SOFTCAP: f32 = 50.0;
/// Keys a query sees: itself and the eleven before it.
const LEFT: usize = 11;

/// Soft tokens for `valid_mel` mel rows: two stride-2 stages, capped.
#[must_use]
pub(crate) fn tokens_for(valid_mel: usize) -> Option<usize> {
    let tokens = valid_mel.div_ceil(2).div_ceil(2);
    (tokens <= MAX_TOKENS).then_some(tokens)
}

/// Whether the container at `path` carries a Gemma 4 audio tower at all.
///
/// Asked before [`Gemma4AudioTower::open`] so that a checkpoint exported
/// without the tower - the text-and-image exports - turns audio input off at
/// boot, rather than failing boot on a tensor that was never meant to be
/// there. A directory answers from its `config.json`, a GGUF from the
/// projector type it declares for audio.
///
/// # Errors
///
/// [`AudioError::Io`] or [`AudioError::Checkpoint`] when the container itself
/// cannot be read: a path that is not there is a mistake, not a missing tower.
pub fn holds_audio_tower(path: &Path) -> Result<bool, AudioError> {
    if path.is_dir() {
        let text =
            std::fs::read_to_string(path.join("config.json")).map_err(|source| AudioError::Io {
                what: "reading config.json",
                source,
            })?;
        let config: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| AudioError::Checkpoint {
                what: format!("config.json: {e}"),
            })?;
        return Ok(config
            .get("audio_config")
            .is_some_and(|block| !block.is_null()));
    }
    let store = crate::audio_ops::store::TensorStore::open_gguf(path)?;
    Ok(store.gguf().is_some_and(|gguf| {
        gguf.metadata().get_str("clip.audio.projector_type") == Some("gemma4a")
    }))
}

/// The Gemma 4 audio tower, loaded.
#[derive(Debug)]
pub struct Gemma4AudioTower {
    weights: AudioWeights,
    mel: MelSpec,
    bytes: u64,
}

impl Gemma4AudioTower {
    /// Opens the tower from the MLX directory or the mmproj GGUF at `path`.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming what is missing, or a device error.
    pub fn open(backend: &Arc<dyn Backend>, path: &Path) -> Result<Self, AudioError> {
        let ops = Ops::new(Arc::clone(backend));
        let weights = AudioWeights::load(&ops, path)?;
        let mel = MelSpec {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 320,
            hop_length: 160,
            n_mels: weights.dims.mels,
            window: Window::HannPeriodic,
            framing: Framing::LeftPadded {
                pad_to_multiple: 128,
            },
            preemphasis: Preemphasis::None,
            spectrum: Spectrum::Magnitude,
            filterbank: Filterbank::Computed {
                scale: MelScale::Htk,
                slaney_norm: false,
                f_min: 0.0,
                f_max: 8_000.0,
            },
            log_scale: LogScale::Natural { guard: 1e-3 },
        };
        let bytes = std::fs::metadata(path).map_or(0, |m| if m.is_file() { m.len() } else { 0 });
        Ok(Self {
            weights,
            mel,
            bytes,
        })
    }

    /// The tower output and the embedded rows of one clip, for parity checks.
    ///
    /// # Errors
    ///
    /// A device error.
    pub fn encode_stages(
        &self,
        backend: &Arc<dyn Backend>,
        clip: &AudioClip,
    ) -> Result<(DeviceBuffer, DeviceBuffer), AudioError> {
        let ops = Ops::new(Arc::clone(backend));
        Encoder {
            ops: &ops,
            w: &self.weights,
        }
        .run(std::slice::from_ref(clip))
    }
}

impl AudioTower for Gemma4AudioTower {
    fn mel_spec(&self) -> &MelSpec {
        &self.mel
    }

    fn tokens_for(&self, valid_mel: usize) -> Option<usize> {
        tokens_for(valid_mel)
    }

    fn max_seconds(&self) -> f64 {
        30.0
    }

    fn out_hidden(&self) -> usize {
        self.weights.dims.text
    }

    fn device_bytes(&self) -> u64 {
        self.bytes
    }

    fn encode(
        &self,
        backend: &Arc<dyn Backend>,
        clips: &[AudioClip],
    ) -> Result<EncodedRows, AudioError> {
        let ops = Ops::new(Arc::clone(backend));
        let encoder = Encoder {
            ops: &ops,
            w: &self.weights,
        };
        let mut grids = Vec::with_capacity(clips.len());
        for clip in clips {
            let n = u32::try_from(clip.tokens()).map_err(|_| AudioError::BadSpec {
                reason: "too many tokens".into(),
            })?;
            grids.push(VisionGrid::new(1, 1, n));
        }
        // Every clip in one pass: the row-wise work - the projections, which
        // are most of the tower - runs once over all their rows, and only
        // the time-mixing steps go clip by clip.
        let (_, rows) = encoder.run(clips)?;
        Ok(EncodedRows { rows, grids })
    }
}

struct Encoder<'a> {
    ops: &'a Ops,
    w: &'a AudioWeights,
}

#[expect(
    clippy::many_single_char_names,
    reason = "the tower's widths and tensors keep their short names: d, n, q, k, v, h, x"
)]
impl Encoder<'_> {
    fn rms(
        &self,
        out: &DeviceBuffer,
        input: &DeviceBuffer,
        weight: Option<&DeviceBuffer>,
        rows: usize,
        dim: usize,
    ) -> Result<(), AudioError> {
        Ok(self.ops.rms_norm(&NormDesc {
            out: out.view(),
            input: input.view(),
            weight: weight.map(DeviceBuffer::view),
            bias: None,
            rows,
            dim,
            eps: EPS,
        })?)
    }

    fn clamp(
        &self,
        x: &DeviceBuffer,
        count: usize,
        (min, max): (f32, f32),
    ) -> Result<(), AudioError> {
        Ok(self.ops.unary(&UnaryDesc {
            op: UnaryOp::Clamp { min, max },
            out: x.view(),
            input: x.view(),
            count,
        })?)
    }

    /// `clamp_out(W * clamp_in(x))`, leaving `x` untouched.
    fn clipped(
        &self,
        c: &Clipped,
        x: &DeviceBuffer,
        rows: usize,
    ) -> Result<DeviceBuffer, AudioError> {
        let (inp, out) = (c.linear.in_dim(), c.linear.out_dim());
        // Clamped into its own scratch, the input itself unchanged.
        let held = self.ops.scratch_uninit(rows * inp)?;
        let (min, max) = c.input;
        self.ops.unary(&UnaryDesc {
            op: UnaryOp::Clamp { min, max },
            out: held.view(),
            input: x.view(),
            count: rows * inp,
        })?;
        let y = self.ops.scratch_uninit(rows * out)?;
        c.linear.forward(self.ops, held.view(), y.view(), rows)?;
        self.clamp(&y, rows * out, c.output)?;
        Ok(y)
    }

    fn feed_forward(&self, f: &FeedForward, x: &DeviceBuffer, n: usize) -> Result<(), AudioError> {
        let d = self.w.dims.d_model;
        let h = self.ops.scratch_uninit(n * d)?;
        self.rms(&h, x, Some(&f.pre), n, d)?;
        let up = self.clipped(&f.up, &h, n)?;
        self.ops.unary(&UnaryDesc {
            op: UnaryOp::Silu,
            out: up.view(),
            input: up.view(),
            count: n * self.w.dims.ff,
        })?;
        let down = self.clipped(&f.down, &up, n)?;
        self.rms(&h, &down, Some(&f.post), n, d)?;
        Ok(self.ops.axpy(&AxpyDesc {
            y: x.view(),
            x: h.view(),
            alpha: 0.5,
            count: n * d,
        })?)
    }

    fn attention(&self, l: &Layer, x: &DeviceBuffer, spans: &[usize]) -> Result<(), AudioError> {
        let n: usize = spans.iter().sum();
        let d = self.w.dims.d_model;
        let heads = self.w.dims.heads;
        let h = self.ops.scratch_uninit(n * d)?;
        self.rms(&h, x, Some(&l.norm_pre_attn), n, d)?;
        let (q, k, v) = (
            self.clipped(&l.q, &h, n)?,
            self.clipped(&l.k, &h, n)?,
            self.clipped(&l.v, &h, n)?,
        );
        let att = self.ops.scratch_uninit(n * d)?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the kernel takes the scale as f32"
        )]
        let k_scale = (1.0f64.exp().ln_1p() / std::f64::consts::LN_2) as f32;
        // Each clip attends within itself, from its own first frame.
        let mut start = 0;
        for &n in spans {
            self.ops.window_attention(&WindowAttnDesc {
                out: rows_of(&att, start, n, d)?,
                q: rows_of(&q, start, n, d)?,
                k: rows_of(&k, start, n, d)?,
                v: rows_of(&v, start, n, d)?,
                n_q: n,
                n_k: n,
                heads,
                head_dim: d / heads,
                window: AttnWindow {
                    q_abs0: 0,
                    k_abs0: 0,
                    k_end: n,
                    chunk: 1,
                    left_chunks: LEFT,
                },
                bias_uv: None,
                q_dim_scale: Some(l.q_dim_scale.view()),
                rel: Some(RelPos {
                    table: l.rel.view(),
                    d_min: 0,
                    rows: LEFT + 2,
                }),
                scale: 1.0,
                ac_scale: k_scale,
                softcap: Some(SOFTCAP),
            })?;
            start += n;
        }
        let o = self.clipped(&l.post, &att, n)?;
        self.rms(&h, &o, Some(&l.norm_post_attn), n, d)?;
        Ok(self.ops.axpy(&AxpyDesc {
            y: x.view(),
            x: h.view(),
            alpha: 1.0,
            count: n * d,
        })?)
    }

    fn light_conv(&self, l: &Layer, x: &DeviceBuffer, spans: &[usize]) -> Result<(), AudioError> {
        let n: usize = spans.iter().sum();
        let d = self.w.dims.d_model;
        let taps = self.w.dims.kernel;
        let h = self.ops.scratch_uninit(n * d)?;
        self.rms(&h, x, Some(&l.conv_pre_norm), n, d)?;
        let wide = self.clipped(&l.conv_start, &h, n)?;
        // Causal: `taps - 1` zero rows ahead of each clip's first frame, so
        // no clip's convolution reaches into the clip before it.
        let padded = self.ops.scratch((taps - 1) * spans.len() * d + n * d)?;
        let (mut start, mut at) = (0, 0);
        for &rows in spans {
            let pad = rows_of(&padded, at, taps - 1 + rows, d)?;
            self.ops.glu(&GluDesc {
                out: rows_of(&padded, at + taps - 1, rows, d)?,
                input: rows_of(&wide, start, rows, 2 * d)?,
                rows,
                channels: d,
            })?;
            self.ops.dwconv1d(&DwConv1dDesc {
                out: rows_of(&h, start, rows, d)?,
                input: pad,
                weight: l.conv_dw.view(),
                rows,
                channels: d,
                taps,
            })?;
            start += rows;
            at += taps - 1 + rows;
        }
        let normed = self.ops.scratch_uninit(n * d)?;
        self.rms(&normed, &h, Some(&l.conv_norm), n, d)?;
        self.ops.unary(&UnaryDesc {
            op: UnaryOp::Silu,
            out: normed.view(),
            input: normed.view(),
            count: n * d,
        })?;
        let out = self.clipped(&l.conv_end, &normed, n)?;
        Ok(self.ops.axpy(&AxpyDesc {
            y: x.view(),
            x: out.view(),
            alpha: 1.0,
            count: n * d,
        })?)
    }

    /// The tower output `[tokens][out]` and the embedded rows `[tokens][text]`.
    /// One clip's subsampling front end and input projection, into `x`.
    fn front(&self, clip: &AudioClip, x: &DeviceView<'_>) -> Result<(), AudioError> {
        let dims = self.w.dims;
        let [c0, c1] = dims.channels;
        let mel = clip.mel();
        let t0 = mel.valid();
        let (t1, t2) = (t0.div_ceil(2), t0.div_ceil(2).div_ceil(2));
        let (f0, f1, f2) = (
            dims.mels,
            dims.mels.div_ceil(2),
            dims.mels.div_ceil(2).div_ceil(2),
        );
        if t2 != clip.tokens() {
            return Err(AudioError::BadSpec {
                reason: format!("{t0} mel rows make {t2} tokens, not {}", clip.tokens()),
            });
        }
        let input = upload_f32(self.ops, &mel.as_slice()[..t0 * f0])?;
        let stage = |input: &DeviceBuffer,
                     i: usize,
                     (t_in, f_in, c_in): (usize, usize, usize),
                     (t_out, f_out, c_out): (usize, usize, usize)|
         -> Result<DeviceBuffer, AudioError> {
            let out = self.ops.scratch(t_out * f_out * c_out)?;
            self.ops.conv2d_s2(&Conv2dDesc {
                out: out.view(),
                input: input.view(),
                weight: self.w.conv[i].view(),
                bias: None,
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
                    batch: 1,
                },
                relu: false,
            })?;
            self.ops.layer_norm(&NormDesc {
                out: out.view(),
                input: out.view(),
                weight: Some(self.w.conv_norm[i].view()),
                bias: None,
                rows: t_out * f_out,
                dim: c_out,
                eps: EPS,
            })?;
            self.ops.unary(&UnaryDesc {
                op: UnaryOp::Relu,
                out: out.view(),
                input: out.view(),
                count: t_out * f_out * c_out,
            })?;
            Ok(out)
        };
        let s1 = stage(&input, 0, (t0, f0, 1), (t1, f1, c0))?;
        let s2 = stage(&s1, 1, (t1, f1, c0), (t2, f2, c1))?;
        self.w.input_proj.forward(self.ops, s2.view(), *x, t2)?;
        Ok(())
    }

    /// Every clip's tower output `[tokens][out]` and embedded rows
    /// `[tokens][text]`, clip after clip.
    fn run(&self, clips: &[AudioClip]) -> Result<(DeviceBuffer, DeviceBuffer), AudioError> {
        let dims = self.w.dims;
        let d = dims.d_model;
        let spans: Vec<usize> = clips.iter().map(AudioClip::tokens).collect();
        let t2: usize = spans.iter().sum();
        let x = self.ops.scratch_uninit(t2 * d)?;
        let mut start = 0;
        for clip in clips {
            self.front(clip, &rows_of(&x, start, clip.tokens(), d)?)?;
            start += clip.tokens();
        }
        for layer in &self.w.layers {
            self.feed_forward(&layer.ff[0], &x, t2)?;
            self.attention(layer, &x, &spans)?;
            self.light_conv(layer, &x, &spans)?;
            self.feed_forward(&layer.ff[1], &x, t2)?;
            self.rms(&x, &x, Some(&layer.norm_out), t2, d)?;
        }
        let hidden = self.ops.scratch_uninit(t2 * dims.out)?;
        self.w
            .output_proj
            .forward(self.ops, x.view(), hidden.view(), t2)?;
        let normed = self.ops.scratch_uninit(t2 * dims.out)?;
        self.rms(&normed, &hidden, None, t2, dims.out)?;
        let embedded = self.ops.scratch_uninit(t2 * dims.text)?;
        self.w
            .embed
            .forward(self.ops, normed.view(), embedded.view(), t2)?;
        Ok((hidden, embedded))
    }
}

/// Rows `start .. start + n` of a `[rows][width]` f32 buffer.
fn rows_of(
    b: &DeviceBuffer,
    start: usize,
    n: usize,
    width: usize,
) -> Result<DeviceView<'_>, AudioError> {
    let bytes = |r: usize| u64::try_from(r * width * 4).unwrap_or(u64::MAX);
    Ok(b.slice(bytes(start), bytes(n))?)
}

#[cfg(test)]
mod tests;
