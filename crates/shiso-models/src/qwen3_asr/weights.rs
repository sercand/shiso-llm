//! Every tensor of the checkpoint on the device, in the layout the kernels read.
//!
//! Linears are imported from the mapping as they are stored: bf16, f16 and
//! `Q8_0` (the HF export and both GGUF pairs) onto the audio GEMM, the MLX
//! export's 8-bit affine words onto the engine's projection path. The mmproj
//! `conv_out` is f16 in one GGUF pair and `Q8_0` in the other; both are dense
//! and load the same way. Everything small - norms, conv kernels, biases - is
//! widened to f32 on the host, and the MLX export's channels-last conv
//! kernels (`[out][f][t][in]`) are permuted to the reference layout on the
//! way, so
//! every export looks the same to the encoder. The conv kernels are then
//! reordered once more into the rows of a GEMM over the conv's patches
//! ([`Conv`]).
//!
//! The output head is the embedding (the config ties them): the MLX
//! export and one GGUF have no separate head, and the others store a
//! bit-identical copy, so it is loaded once.

use shiso_audio::AudioError;
use shiso_core::DeviceBuffer;

use super::config::{Flavour, Located, Qwen3AsrConfig, Stores, TOKENS_PER_CHUNK};
use super::names::{Names, Stem};
use crate::audio_ops::load::{LinearSource, conv_weight, f32_param, linear, upload_f32};
use crate::audio_ops::store::TensorStore;
use crate::audio_ops::{GemmType, Linear, LinearWeight, Ops};

/// A layer norm's parameters.
#[derive(Debug)]
pub struct Norm {
    /// Scale, `[dim]`.
    pub weight: DeviceBuffer,
    /// Offset, `[dim]`.
    pub bias: DeviceBuffer,
}

/// One conv of the stem, 3x3, stride 2, padding 1, as a linear over its
/// patches: an f32 `[out][3 time][3 frequency][in]` weight on the audio GEMM,
/// with the bias.
///
/// The reference's kernel is `[out][in][frequency][time]`, because its conv
/// runs over `[mels][frames]` images; the audio kernels take
/// channels-last `[t][f][c]` images, time first.
#[derive(Debug)]
pub struct Conv {
    /// `in_dim` is `9 * channels in`.
    pub linear: Linear,
}

/// One pre-norm transformer layer of the tower.
#[derive(Debug)]
pub struct TowerLayer {
    /// Norm before attention.
    pub attn_norm: Norm,
    /// Query projection, with bias.
    pub q: Linear,
    /// Key projection, with bias.
    pub k: Linear,
    /// Value projection, with bias.
    pub v: Linear,
    /// Output projection, with bias.
    pub o: Linear,
    /// Norm before the feed-forward.
    pub ffn_norm: Norm,
    /// Feed-forward up, with bias.
    pub fc1: Linear,
    /// Feed-forward down, with bias.
    pub fc2: Linear,
}

/// The audio tower on the device.
#[derive(Debug)]
pub struct TowerWeights {
    /// The three stem convs, `1 -> C -> C -> C`.
    pub convs: [Conv; 3],
    /// The flattened stem into the tower's width, `C * F -> d_model`, no bias.
    pub conv_out: Linear,
    /// The sinusoid rows added after `conv_out`, `[13][d_model]` f32: only
    /// the first 13 are ever read, because positions restart in every chunk.
    pub positions: DeviceBuffer,
    /// The transformer layers.
    pub layers: Vec<TowerLayer>,
    /// The closing norm.
    pub ln_post: Norm,
    /// Output head, `d_model -> d_model` with bias.
    pub proj1: Linear,
    /// Output head, `d_model -> output_dim` with bias.
    pub proj2: Linear,
}

/// One decoder layer.
#[derive(Debug)]
pub struct TextLayer {
    /// RMS norm before attention, `[hidden]`.
    pub attn_norm: DeviceBuffer,
    /// Query projection, no bias.
    pub q: Linear,
    /// Key projection, no bias.
    pub k: Linear,
    /// Value projection, no bias.
    pub v: Linear,
    /// Output projection, no bias.
    pub o: Linear,
    /// Per-head RMS norm on the queries, `[head_dim]`.
    pub q_norm: DeviceBuffer,
    /// Per-head RMS norm on the keys, `[head_dim]`.
    pub k_norm: DeviceBuffer,
    /// RMS norm before the MLP, `[hidden]`.
    pub ffn_norm: DeviceBuffer,
    /// MLP gate, no bias.
    pub gate: Linear,
    /// MLP up, no bias.
    pub up: Linear,
    /// MLP down, no bias.
    pub down: Linear,
}

/// The decoder on the device.
#[derive(Debug)]
pub struct TextWeights {
    /// The token embedding, `[vocab][hidden]`; also the output head.
    pub embed: Linear,
    /// The layers.
    pub layers: Vec<TextLayer>,
    /// The final RMS norm, `[hidden]`.
    pub norm: DeviceBuffer,
}

/// The whole checkpoint on the device.
#[derive(Debug)]
pub struct QwenAsrWeights {
    /// The audio tower.
    pub tower: TowerWeights,
    /// The decoder.
    pub text: TextWeights,
}

/// The tower's position rows: `rows` of `width`, the first half of each
/// `sin(p * w_i)` and the second half `cos(p * w_i)` (not interleaved), with
/// `w_i = exp(-i * ln(10000) / (width/2 - 1))`, as the reference encoder
/// builds its table.
pub(crate) fn sinusoids(rows: usize, width: usize) -> Vec<f32> {
    let half = width / 2;
    let step = 10_000f64.ln() / crate::audio_ops::f64_of(half.saturating_sub(1).max(1));
    let mut out = vec![0.0f32; rows * width];
    for (p, row) in out.chunks_exact_mut(width).enumerate() {
        for i in 0..half {
            let angle = crate::audio_ops::f64_of(p) * (-step * crate::audio_ops::f64_of(i)).exp();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the table is f32 by contract"
            )]
            {
                row[i] = angle.sin() as f32;
                row[half + i] = angle.cos() as f32;
            }
        }
    }
    out
}

/// A reference `[out][in][3 frequency][3 time]` kernel as the patch GEMM's
/// `[out][3 time][3 frequency][in]` rows (see [`Conv`]).
fn patch_rows(values: &[f32], out: usize, inp: usize) -> Vec<f32> {
    let mut rows = vec![0.0f32; values.len()];
    for o in 0..out {
        for i in 0..inp {
            for kf in 0..3 {
                for kt in 0..3 {
                    rows[(o * 9 + kt * 3 + kf) * inp + i] =
                        values[((o * inp + i) * 3 + kf) * 3 + kt];
                }
            }
        }
    }
    rows
}

struct Loader<'a> {
    ops: &'a Ops,
    store: &'a TensorStore,
    names: Names,
    group: Option<usize>,
}

impl Loader<'_> {
    fn lin(&self, role: Stem, out: usize, inp: usize, bias: bool) -> Result<Linear, AudioError> {
        let (w, b) = (self.names.weight(role), self.names.bias(role));
        let src = LinearSource {
            weight: &w,
            bias: bias.then_some(b.as_str()),
            affine_group: self.group,
        };
        linear(self.ops, self.store, &src, out, inp)
    }

    fn param(&self, name: &str, dim: usize) -> Result<DeviceBuffer, AudioError> {
        Ok(f32_param(self.ops, self.store, name, dim)?.0)
    }

    fn norm(&self, role: Stem, dim: usize) -> Result<Norm, AudioError> {
        Ok(Norm {
            weight: self.param(&self.names.weight(role), dim)?,
            bias: self.param(&self.names.bias(role), dim)?,
        })
    }

    fn rms(&self, role: Stem, dim: usize) -> Result<DeviceBuffer, AudioError> {
        self.param(&self.names.weight(role), dim)
    }

    fn conv(&self, i: u8, out: usize, inp: usize, channels_last: bool) -> Result<Conv, AudioError> {
        let role = Stem::Conv(i);
        let values = conv_weight(
            self.store,
            &self.names.weight(role),
            &[out, inp, 3, 3],
            channels_last,
        )?;
        Ok(Conv {
            linear: Linear {
                weight: LinearWeight::Gemm {
                    dtype: GemmType::F32,
                    buffer: upload_f32(self.ops, &patch_rows(&values, out, inp))?,
                    in_dim: 9 * inp,
                    out_dim: out,
                },
                bias: Some(self.param(&self.names.bias(role), out)?),
            },
        })
    }
}

impl QwenAsrWeights {
    /// Loads every tensor of `located` onto `ops`' device.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming the export and the first tensor that
    /// is missing, does not have the shape `config` implies, or is stored in
    /// an encoding no multiply here reads; a device error.
    pub fn load(
        ops: &Ops,
        located: &Located,
        config: &Qwen3AsrConfig,
        stores: &Stores,
    ) -> Result<Self, AudioError> {
        let names = Names::new(located.flavour);
        let group = located.affine_group();
        let tower = Loader {
            ops,
            store: stores.tower(),
            names,
            group,
        };
        let text = Loader {
            ops,
            store: stores.text(),
            names,
            group,
        };
        let loaded = load_tower(&tower, config, located.flavour == Flavour::Mlx)
            .and_then(|tower| Ok((tower, load_text(&text, config)?)));
        match loaded {
            Ok((tower, text)) => Ok(Self { tower, text }),
            Err(AudioError::Checkpoint { what }) => Err(AudioError::Checkpoint {
                what: format!("{:?} export: {what}", located.flavour),
            }),
            Err(other) => Err(other),
        }
    }
}

fn load_tower(
    l: &Loader<'_>,
    config: &Qwen3AsrConfig,
    channels_last: bool,
) -> Result<TowerWeights, AudioError> {
    let a = &config.audio;
    let (c, d) = (a.conv_channels, a.d_model);
    // Three stride-2, pad-1, 3-tap convs over the mel axis.
    let halve = |x: usize| x.saturating_sub(1) / 2 + 1;
    let freq = halve(halve(halve(a.mels)));
    let convs = [
        l.conv(1, c, 1, channels_last)?,
        l.conv(2, c, c, channels_last)?,
        l.conv(3, c, c, channels_last)?,
    ];
    let conv_out = l.lin(Stem::ConvOut, d, c * freq, false)?;
    let positions = upload_f32(l.ops, &sinusoids(TOKENS_PER_CHUNK, d))?;
    let mut layers = Vec::with_capacity(a.layers);
    for n in 0..a.layers {
        layers.push(TowerLayer {
            attn_norm: l.norm(Stem::AttnNorm(n), d)?,
            q: l.lin(Stem::Q(n), d, d, true)?,
            k: l.lin(Stem::K(n), d, d, true)?,
            v: l.lin(Stem::V(n), d, d, true)?,
            o: l.lin(Stem::O(n), d, d, true)?,
            ffn_norm: l.norm(Stem::FfnNorm(n), d)?,
            fc1: l.lin(Stem::Fc1(n), a.ffn, d, true)?,
            fc2: l.lin(Stem::Fc2(n), d, a.ffn, true)?,
        });
    }
    Ok(TowerWeights {
        convs,
        conv_out,
        positions,
        layers,
        ln_post: l.norm(Stem::PostLn, d)?,
        proj1: l.lin(Stem::Proj1, d, d, true)?,
        proj2: l.lin(Stem::Proj2, a.output_dim, d, true)?,
    })
}

fn load_text(l: &Loader<'_>, config: &Qwen3AsrConfig) -> Result<TextWeights, AudioError> {
    let t = &config.text;
    let h = t.hidden;
    let (q_out, kv_out) = (t.heads * t.head_dim, t.kv_heads * t.head_dim);
    let mut layers = Vec::with_capacity(t.layers);
    for n in 0..t.layers {
        layers.push(TextLayer {
            attn_norm: l.rms(Stem::DecAttnNorm(n), h)?,
            q: l.lin(Stem::DecQ(n), q_out, h, false)?,
            k: l.lin(Stem::DecK(n), kv_out, h, false)?,
            v: l.lin(Stem::DecV(n), kv_out, h, false)?,
            o: l.lin(Stem::DecO(n), h, q_out, false)?,
            q_norm: l.rms(Stem::QNorm(n), t.head_dim)?,
            k_norm: l.rms(Stem::KNorm(n), t.head_dim)?,
            ffn_norm: l.rms(Stem::DecFfnNorm(n), h)?,
            gate: l.lin(Stem::Gate(n), t.ffn, h, false)?,
            up: l.lin(Stem::Up(n), t.ffn, h, false)?,
            down: l.lin(Stem::Down(n), h, t.ffn, false)?,
        });
    }
    Ok(TextWeights {
        embed: l.lin(Stem::Embed, t.vocab, h, false)?,
        layers,
        norm: l.rms(Stem::Norm, h)?,
    })
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
