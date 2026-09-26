//! Every tensor of the checkpoint on the device, in the layout the kernels read.
//!
//! Linears are imported from the mapping as they are stored; a fused
//! query-key-value matrix is imported as three row ranges of itself. Norms and
//! biases are widened to f32 on the host. The one re-laid-out tensor is the
//! upsampling conv, which runs as an unfold and a GEMM and so wants its taps
//! outermost within each output row.

use shiso_audio::AudioError;
use shiso_core::DeviceBuffer;

use super::config::{DiarConfig, Located};
use super::names::{Qkv, Stem};
use crate::audio_ops::load::{LinearSource, f32_param, linear, linear_rows, upload_f32};
use crate::audio_ops::store::{TensorStore, dequant};
use crate::audio_ops::{GemmType, Linear, LinearWeight, Ops};

/// Taps of the upsampling conv.
pub const UPSAMPLE_TAPS: usize = 3;

/// A layer norm's parameters.
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
    /// Query projection, no bias.
    pub q: Linear,
    /// Key projection, no bias.
    pub k: Linear,
    /// Value projection, no bias.
    pub v: Linear,
    /// Output projection, with bias.
    pub o: Linear,
    /// Norm before the feed-forward.
    pub norm2: Norm,
    /// Feed-forward up, with bias.
    pub fc1: Linear,
    /// Feed-forward down, with bias.
    pub fc2: Linear,
}

/// The whole checkpoint on the device.
#[derive(Debug)]
pub struct DiarWeights {
    /// Stacked mel frames to the encoder width, no bias.
    pub proj: Linear,
    /// Norm before the first layer.
    pub in_norm: Norm,
    /// The layers.
    pub layers: Vec<Layer>,
    /// Norm after the last layer.
    pub final_norm: Norm,
    /// Encoder width to the head width, with bias.
    pub enc_proj: Linear,
    /// The upsampling conv as a GEMM over unfolded rows: `[stack·head][taps·head]`,
    /// taps outermost within a row, with bias.
    pub upsample: Linear,
    /// The head's hidden linear, with bias.
    pub dense: Linear,
    /// The speaker logits, with bias.
    pub out: Linear,
    /// The learned silence embedding, host side: the speaker cache pads with it.
    pub silence: Vec<f32>,
}

fn checkpoint(what: String) -> AudioError {
    AudioError::Checkpoint { what }
}

/// A conv weight `[out][in][taps]` (or `[out][taps][in]` when `taps_first`)
/// as the row-major matrix `[out][taps·in]` with taps outermost, which is what
/// a GEMM over unfolded rows multiplies by.
#[must_use]
pub fn upsample_matrix(
    values: &[f32],
    out: usize,
    inp: usize,
    taps: usize,
    taps_first: bool,
) -> Vec<f32> {
    if taps_first {
        return values.to_vec();
    }
    let mut matrix = vec![0.0f32; out * taps * inp];
    for o in 0..out {
        for i in 0..inp {
            for t in 0..taps {
                matrix[(o * taps + t) * inp + i] = values[(o * inp + i) * taps + t];
            }
        }
    }
    matrix
}

impl DiarWeights {
    /// Loads every tensor `config` implies.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming the tensor and the export for a
    /// missing or misshapen tensor; a device error while importing.
    pub fn load(
        ops: &Ops,
        located: &Located,
        config: &DiarConfig,
        store: &TensorStore,
    ) -> Result<Self, AudioError> {
        Self::load_all(ops, located, config, store).map_err(|e| match e {
            AudioError::Checkpoint { what } => {
                checkpoint(format!("{:?} export: {what}", located.flavour))
            }
            other => other,
        })
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "the widths keep the short names the model math uses: c, d, h"
    )]
    fn load_all(
        ops: &Ops,
        located: &Located,
        c: &DiarConfig,
        store: &TensorStore,
    ) -> Result<Self, AudioError> {
        let names = located.names();
        let d = c.d_model;
        let norm = |stem: Stem| -> Result<Norm, AudioError> {
            Ok(Norm {
                weight: f32_param(ops, store, &names.weight(stem), d)?.0,
                bias: f32_param(ops, store, &names.bias(stem), d)?.0,
            })
        };
        let lin = |stem: Stem, out: usize, inp: usize, bias: bool| -> Result<Linear, AudioError> {
            let weight = names.weight(stem);
            let bias_name = names.bias(stem);
            linear(
                ops,
                store,
                &LinearSource {
                    weight: &weight,
                    bias: bias.then_some(bias_name.as_str()),
                    affine_group: None,
                },
                out,
                inp,
            )
        };
        let mut layers = Vec::with_capacity(c.layers);
        for n in 0..c.layers {
            let [q, k, v] = match names.qkv(n) {
                Qkv::Split(split) => {
                    let one = |name: &String| {
                        linear(
                            ops,
                            store,
                            &LinearSource {
                                weight: name,
                                bias: None,
                                affine_group: None,
                            },
                            d,
                            d,
                        )
                    };
                    [one(&split[0])?, one(&split[1])?, one(&split[2])?]
                }
                Qkv::Fused(name) => [
                    linear_rows(ops, store, &name, 0..d, 3 * d, d)?,
                    linear_rows(ops, store, &name, d..2 * d, 3 * d, d)?,
                    linear_rows(ops, store, &name, 2 * d..3 * d, 3 * d, d)?,
                ],
            };
            layers.push(Layer {
                norm1: norm(Stem::Norm1(n))?,
                q,
                k,
                v,
                o: lin(Stem::O(n), d, d, true)?,
                norm2: norm(Stem::Norm2(n))?,
                fc1: lin(Stem::Fc1(n), c.ff, d, true)?,
                fc2: lin(Stem::Fc2(n), d, c.ff, true)?,
            });
        }
        let h = c.head_hidden;
        let up_out = c.stack * h;
        let up_name = names.weight(Stem::Upsample);
        let up = dequant(&store.require(&up_name)?)?;
        if up.len() != up_out * h * UPSAMPLE_TAPS {
            return Err(checkpoint(format!(
                "{up_name} holds {} values, not {up_out} x {h} x {UPSAMPLE_TAPS}",
                up.len()
            )));
        }
        let matrix = upsample_matrix(&up, up_out, h, UPSAMPLE_TAPS, names.upsample_taps_first());
        let upsample = Linear {
            weight: LinearWeight::Gemm {
                dtype: GemmType::F32,
                buffer: upload_f32(ops, &matrix)?,
                in_dim: UPSAMPLE_TAPS * h,
                out_dim: up_out,
            },
            bias: Some(f32_param(ops, store, &names.bias(Stem::Upsample), up_out)?.0),
        };
        let silence = dequant(&store.require(names.silence())?)?;
        if silence.len() != d {
            return Err(checkpoint(format!(
                "{} holds {} values, not {d}",
                names.silence(),
                silence.len()
            )));
        }
        Ok(Self {
            proj: lin(Stem::Proj, d, c.stack * c.mels, false)?,
            in_norm: norm(Stem::InNorm)?,
            layers,
            final_norm: norm(Stem::FinalNorm)?,
            enc_proj: lin(Stem::EncProj, h, d, true)?,
            upsample,
            dense: lin(Stem::Dense, h, h, true)?,
            out: lin(Stem::Out, c.speakers, h, true)?,
            silence,
        })
    }
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
