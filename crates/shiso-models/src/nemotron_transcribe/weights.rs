//! Every tensor of the encoder export on the device, in the layout the
//! kernels read.
//!
//! The three transformers share one spelling below their prefix, so one
//! loader reads all of them; a fused query-key-value matrix is imported as
//! three row ranges of itself. Norms and biases are widened to f32. The
//! speaker head's upsampling conv is re-laid out once, as a GEMM over
//! unfolded rows, and the fusion's per-speaker parameters stay on the host:
//! they are eight values a row.

use shiso_audio::AudioError;
use shiso_core::DeviceBuffer;

use super::config::{BlockDims, PeeConfig};
use crate::audio_ops::load::{LinearSource, f32_param, linear, linear_rows, upload_f32};
use crate::audio_ops::store::{TensorStore, dequant};
use crate::audio_ops::{GemmType, Linear, LinearWeight, Ops};
use crate::nemotron_diar::weights::{UPSAMPLE_TAPS, upsample_matrix};

/// A layer norm's parameters, on the device.
#[derive(Debug)]
pub struct Norm {
    /// Scale.
    pub weight: DeviceBuffer,
    /// Offset.
    pub bias: DeviceBuffer,
}

/// One transformer layer.
#[derive(Debug)]
pub struct Layer {
    /// Norm before attention.
    pub norm1: Norm,
    /// Query projection.
    pub q: Linear,
    /// Key projection.
    pub k: Linear,
    /// Value projection.
    pub v: Linear,
    /// The per-head query and key norms, when the block norms them.
    pub qk_norm: Option<(Norm, Norm)>,
    /// Output projection, with bias.
    pub o: Linear,
    /// Norm before the feed-forward.
    pub norm2: Norm,
    /// Feed-forward up, with bias.
    pub fc1: Linear,
    /// Feed-forward down, with bias.
    pub fc2: Linear,
}

/// One rotary transformer: its layers between an input and an output norm.
#[derive(Debug)]
pub struct Transformer {
    /// Its geometry.
    pub dims: BlockDims,
    /// Norm before the first layer.
    pub embed_norm: Norm,
    /// The layers.
    pub layers: Vec<Layer>,
    /// Norm after the last layer.
    pub final_norm: Norm,
}

/// The speaker branch's head and the parameters its streaming state needs.
#[derive(Debug)]
pub struct SpeakerHead {
    /// Encoder width to the head width, with bias.
    pub enc_proj: Linear,
    /// The upsampling conv as a GEMM over unfolded rows, with bias.
    pub upsample: Linear,
    /// The head's hidden linear, with bias.
    pub dense: Linear,
    /// The speaker logits, with bias.
    pub out: Linear,
    /// The learned silence embedding a compressed cache pads with.
    pub silence: Vec<f32>,
}

/// The fusion of speaker activity into the speech rows.
#[derive(Debug)]
pub struct Fusion {
    /// Norm over the speech rows.
    pub asr_norm: Norm,
    /// Norm over the eight activities, host side: `(weight, bias)`.
    pub diar_norm: (Vec<f32>, Vec<f32>),
    /// `[speakers][d_model]`, host side.
    pub kernel: Vec<f32>,
    /// The fused rows to the decoder's width, with bias.
    pub proj: Linear,
}

/// The timestamp head.
#[derive(Debug)]
pub struct CtcHead {
    /// `d_model -> upsample · d_model`, with bias.
    pub upsample: Linear,
    /// Its transformer.
    pub transformer: Transformer,
    /// The piece logits, with bias.
    pub classifier: Linear,
}

/// The whole export on the device.
#[derive(Debug)]
pub struct PeeWeights {
    /// Stacked mel frames to the speech width, no bias.
    pub asr_proj: Linear,
    /// The speech transformer.
    pub asr: Transformer,
    /// Stacked mel frames to the speaker width, no bias.
    pub diar_proj: Linear,
    /// The speaker transformer.
    pub diar: Transformer,
    /// The speaker head.
    pub head: SpeakerHead,
    /// The fusion.
    pub fusion: Fusion,
    /// The timestamp head.
    pub ctc: CtcHead,
    /// The mel filterbank, `[mels][n_fft / 2 + 1]`.
    pub filterbank: Vec<f32>,
}

fn checkpoint(what: String) -> AudioError {
    AudioError::Checkpoint { what }
}

/// The loader's handles, so the helpers below take one argument.
struct Loader<'a> {
    ops: &'a Ops,
    store: &'a TensorStore,
}

impl Loader<'_> {
    fn host(&self, name: &str, elements: usize) -> Result<Vec<f32>, AudioError> {
        let values = dequant(&self.store.require(name)?)?;
        if values.len() != elements {
            return Err(checkpoint(format!(
                "{name} holds {} values, not {elements}",
                values.len()
            )));
        }
        Ok(values)
    }

    fn norm(&self, stem: &str, dim: usize) -> Result<Norm, AudioError> {
        Ok(Norm {
            weight: f32_param(self.ops, self.store, &format!("{stem}.weight"), dim)?.0,
            bias: f32_param(self.ops, self.store, &format!("{stem}.bias"), dim)?.0,
        })
    }

    fn linear(&self, stem: &str, out: usize, inp: usize, bias: bool) -> Result<Linear, AudioError> {
        let weight = format!("{stem}.weight");
        let bias_name = format!("{stem}.bias");
        linear(
            self.ops,
            self.store,
            &LinearSource {
                weight: &weight,
                bias: bias.then_some(bias_name.as_str()),
                affine_group: None,
            },
            out,
            inp,
        )
    }

    fn transformer(&self, prefix: &str, dims: BlockDims) -> Result<Transformer, AudioError> {
        let d = dims.d_model;
        let mut layers = Vec::with_capacity(dims.layers);
        for n in 0..dims.layers {
            let at = format!("{prefix}.layers.{n}");
            let qkv = format!("{at}.attn.w_qkv.weight");
            let qk_norm = if dims.qk_norm {
                Some((
                    self.norm(&format!("{at}.attn.q_norm"), dims.head_dim)?,
                    self.norm(&format!("{at}.attn.k_norm"), dims.head_dim)?,
                ))
            } else {
                None
            };
            layers.push(Layer {
                norm1: self.norm(&format!("{at}.norm1"), d)?,
                q: linear_rows(self.ops, self.store, &qkv, 0..d, 3 * d, d)?,
                k: linear_rows(self.ops, self.store, &qkv, d..2 * d, 3 * d, d)?,
                v: linear_rows(self.ops, self.store, &qkv, 2 * d..3 * d, 3 * d, d)?,
                qk_norm,
                o: self.linear(&format!("{at}.attn.out_proj"), d, d, true)?,
                norm2: self.norm(&format!("{at}.norm2"), d)?,
                fc1: self.linear(&format!("{at}.ffn.net.0"), dims.ff, d, true)?,
                fc2: self.linear(&format!("{at}.ffn.net.3"), d, dims.ff, true)?,
            });
        }
        Ok(Transformer {
            dims,
            embed_norm: self.norm(&format!("{prefix}.embed_norm"), d)?,
            layers,
            final_norm: self.norm(&format!("{prefix}.final_norm"), d)?,
        })
    }

    fn head(&self, c: &PeeConfig) -> Result<SpeakerHead, AudioError> {
        let h = c.head_hidden;
        let up_out = c.stack * h;
        let up_name = "diar.head.subpixel_upsample.weight";
        let up = self.host(up_name, up_out * h * UPSAMPLE_TAPS)?;
        // Stored `[out][in][taps]`, taps innermost.
        let matrix = upsample_matrix(&up, up_out, h, UPSAMPLE_TAPS, false);
        Ok(SpeakerHead {
            enc_proj: self.linear("diar.head.encoder_proj", h, c.diar.d_model, true)?,
            upsample: Linear {
                weight: LinearWeight::Gemm {
                    dtype: GemmType::F32,
                    buffer: upload_f32(self.ops, &matrix)?,
                    in_dim: UPSAMPLE_TAPS * h,
                    out_dim: up_out,
                },
                bias: Some(
                    f32_param(
                        self.ops,
                        self.store,
                        "diar.head.subpixel_upsample.bias",
                        up_out,
                    )?
                    .0,
                ),
            },
            dense: self.linear("diar.head.first_hidden_to_hidden", h, h, true)?,
            out: self.linear("diar.head.single_hidden_to_spks", c.speakers, h, true)?,
            silence: self.host("diar.head.learnable_sil_emb", c.diar.d_model)?,
        })
    }
}

impl PeeWeights {
    /// Loads every tensor `config` implies out of `store`.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming a missing or misshapen tensor; a
    /// device error while importing.
    pub fn load(ops: &Ops, store: &TensorStore, c: &PeeConfig) -> Result<Self, AudioError> {
        let l = Loader { ops, store };
        let stacked = c.stack * c.mels;
        let (a, s) = (c.asr.d_model, c.speakers);
        Ok(Self {
            asr_proj: l.linear("asr.pre_encode.proj", a, stacked, false)?,
            asr: l.transformer("asr", c.asr)?,
            diar_proj: l.linear("diar.pre_encode.proj", c.diar.d_model, stacked, false)?,
            diar: l.transformer("diar", c.diar)?,
            head: l.head(c)?,
            fusion: Fusion {
                asr_norm: l.norm("fusion.asr_norm", a)?,
                diar_norm: (
                    l.host("fusion.diar_norm.weight", s)?,
                    l.host("fusion.diar_norm.bias", s)?,
                ),
                kernel: l.host("fusion.diar_kernel", s * a)?,
                proj: l.linear("fusion.proj", c.fusion_dim, a, true)?,
            },
            ctc: CtcHead {
                upsample: l.linear("ctc.upsample", c.ctc_upsample * a, a, true)?,
                transformer: l.transformer("ctc.transformer", c.ctc)?,
                classifier: l.linear("ctc.classifier", c.ctc_vocab, c.ctc.d_model, true)?,
            },
            filterbank: l.host("preprocessor.fb", c.mels * (c.n_fft / 2 + 1))?,
        })
    }
}
