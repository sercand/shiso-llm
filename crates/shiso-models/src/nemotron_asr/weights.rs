//! Every tensor of the checkpoint on the device, in the layout the kernels read.
//!
//! Linears are imported from the mapping as they are stored. Everything small -
//! norms, conv kernels, position biases, the LSTM's folded bias - is widened to
//! f32 on the host first, which also makes every export look the same to the
//! encoder. Two tables are built here rather than stored: the relative-position
//! keys each layer attends with, and the per-language bias the prompt folds to.

use shiso_audio::AudioError;
use shiso_core::DeviceBuffer;

use super::config::{AsrConfig, Flavour, Located};
use super::names::{LstmBias, Names, Stem};
use crate::audio_ops::load::{LinearSource, conv_weight, f32_param, linear, upload_f32};
use crate::audio_ops::store::{StoredType, TensorStore, dequant, dequant_affine};
use crate::audio_ops::{GemmType, Linear, LinearWeight, Ops};

/// A layer norm's parameters.
#[derive(Debug)]
pub struct Norm {
    /// Scale.
    pub weight: DeviceBuffer,
    /// Offset.
    pub bias: DeviceBuffer,
}

/// One subsampling stage after the first: depthwise conv, then pointwise.
#[derive(Debug)]
pub struct SubStage {
    /// `[C][1][3][3]`.
    pub dw: DeviceBuffer,
    /// `[C]`.
    pub dw_bias: DeviceBuffer,
    /// Pointwise, `C -> C` with bias.
    pub pw: Linear,
}

/// One conformer block.
#[derive(Debug)]
pub struct Block {
    /// Norm before feed-forward 1.
    pub norm_ff1: Norm,
    /// Feed-forward 1, up and down.
    pub ff1: (Linear, Linear),
    /// Norm before attention.
    pub norm_att: Norm,
    /// Query projection.
    pub q: Linear,
    /// Key projection.
    pub k: Linear,
    /// Value projection.
    pub v: Linear,
    /// Output projection.
    pub o: Linear,
    /// Content bias, `[heads * head_dim]`.
    pub bias_u: DeviceBuffer,
    /// Position bias, `[heads * head_dim]`.
    pub bias_v: DeviceBuffer,
    /// Relative-position keys, `[rel_rows][d_model]`, row `r` for distance
    /// `r + rel_d_min`.
    pub rel: DeviceBuffer,
    /// Norm before the conv module.
    pub norm_conv: Norm,
    /// Pointwise conv to the GLU, `d_model -> 2 d_model`.
    pub pw1: Linear,
    /// Depthwise conv, `[d_model][kernel]`.
    pub dw: DeviceBuffer,
    /// Norm after the depthwise conv.
    pub conv_norm: Norm,
    /// Pointwise conv back, `d_model -> d_model`.
    pub pw2: Linear,
    /// Norm before feed-forward 2.
    pub norm_ff2: Norm,
    /// Feed-forward 2, up and down.
    pub ff2: (Linear, Linear),
    /// The block's closing norm.
    pub norm_out: Norm,
}

/// One LSTM layer.
#[derive(Debug)]
pub struct LstmLayer {
    /// Input weights, with `b_ih + b_hh` as their bias.
    pub w_ih: Linear,
    /// Recurrent weights, without a bias.
    pub w_hh: Linear,
}

/// The whole checkpoint on the device.
#[derive(Debug)]
pub struct AsrWeights {
    /// First subsampling conv, `[C][1][3][3]`.
    pub conv0: DeviceBuffer,
    /// Its bias, `[C]`.
    pub conv0_bias: DeviceBuffer,
    /// Stages 1 and 2.
    pub stages: [SubStage; 2],
    /// Subsampling output, `C * F3 -> d_model`.
    pub sub_out: Linear,
    /// Conformer blocks.
    pub layers: Vec<Block>,
    /// Distance of relative-position row 0.
    pub rel_d_min: i32,
    /// Relative-position rows per layer.
    pub rel_rows: usize,
    /// Prompt linear 1 over the encoder columns only, `d_model -> prompt_hidden`.
    pub prompt_w1: Linear,
    /// Prompt linear 1's bias, host side.
    pub prompt_b1: Vec<f32>,
    /// Prompt linear 1's language columns, `[prompts][prompt_hidden]`, host side.
    pub prompt_cols: Vec<f32>,
    /// Prompt linear 2, `prompt_hidden -> d_model` with bias.
    pub prompt_w2: Linear,
    /// Encoder projection into the joint, with bias.
    pub enc_proj: Linear,
    /// Prediction network embedding, `[vocab][pred_hidden]` f32.
    pub embed: DeviceBuffer,
    /// LSTM layers.
    pub lstm: Vec<LstmLayer>,
    /// Prediction output projection, with bias.
    pub dec_proj: Linear,
    /// Joint output layer, `joint -> vocab` with bias.
    pub head: Linear,
}

fn checkpoint(what: String) -> AudioError {
    AudioError::Checkpoint { what }
}

struct Loader<'a> {
    ops: &'a Ops,
    store: &'a TensorStore,
    names: Names,
    mlx: bool,
    /// Elements per affine group, for an export whose linears are packed words.
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

    fn norm(&self, role: Stem, dim: usize) -> Result<Norm, AudioError> {
        Ok(Norm {
            weight: f32_param(self.ops, self.store, &self.names.weight(role), dim)?.0,
            bias: f32_param(self.ops, self.store, &self.names.bias(role), dim)?.0,
        })
    }

    fn conv(&self, role: Stem, shape: &[usize]) -> Result<DeviceBuffer, AudioError> {
        let values = conv_weight(self.store, &self.names.weight(role), shape, self.mlx)?;
        upload_f32(self.ops, &values)
    }

    fn host_matrix(&self, role: Stem, out: usize, inp: usize) -> Result<Vec<f32>, AudioError> {
        let name = self.names.weight(role);
        let t = self.store.require(&name)?;
        let values = if t.ty == StoredType::U32 {
            let stem = self.names.stem(role);
            dequant_affine(
                &t,
                &self.store.require(&format!("{stem}.scales"))?,
                &self.store.require(&format!("{stem}.biases"))?,
                bits_of(&t, inp)?,
                self.group.ok_or_else(|| {
                    checkpoint(format!(
                        "{name} is packed words but the export states no group"
                    ))
                })?,
            )?
        } else {
            dequant(&t)?
        };
        if values.len() != out * inp {
            return Err(checkpoint(format!(
                "{name} holds {} values, not {out} x {inp}",
                values.len()
            )));
        }
        Ok(values)
    }
}

fn bits_of(t: &crate::audio_ops::store::Stored<'_>, inp: usize) -> Result<u32, AudioError> {
    let words = t.shape.get(1).copied().unwrap_or(0);
    u32::try_from(words * 32 / u64::try_from(inp.max(1)).unwrap_or(1))
        .map_err(|_| checkpoint(format!("{}: bits", t.name)))
}

/// The interleaved sinusoid table of distance `d`: `pe[2m] = sin(d w_m)`,
/// `pe[2m+1] = cos(d w_m)`, `w_m = 10000^(-2m/width)`.
fn sinusoid(d: i32, width: usize) -> Vec<f32> {
    let mut row = vec![0.0f32; width];
    for m in 0..width / 2 {
        let freq =
            10_000f64.powf(-2.0 * crate::audio_ops::f64_of(m) / crate::audio_ops::f64_of(width));
        let angle = f64::from(d) * freq;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the table is f32 by contract"
        )]
        {
            row[2 * m] = angle.sin() as f32;
            row[2 * m + 1] = angle.cos() as f32;
        }
    }
    row
}

impl AsrWeights {
    /// Loads every tensor of `located` onto `ops`' device.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming the first tensor that is missing or
    /// does not have the shape `config` implies; a device error.
    pub fn load(
        ops: &Ops,
        located: &Located,
        config: &AsrConfig,
        store: &TensorStore,
    ) -> Result<Self, AudioError> {
        Self::load_inner(ops, located, config, store).map_err(|e| match e {
            AudioError::Checkpoint { what } => AudioError::Checkpoint {
                what: format!("{:?} export: {what}", located.flavour),
            },
            other => other,
        })
    }

    #[expect(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        clippy::similar_names,
        reason = "one statement per tensor role is the whole loader, and the dimensions keep the short names the model's description gives them (C, d, h, V; w_ih and w_hh)"
    )]
    fn load_inner(
        ops: &Ops,
        located: &Located,
        config: &AsrConfig,
        store: &TensorStore,
    ) -> Result<Self, AudioError> {
        let l = Loader {
            ops,
            store,
            names: Names::new(located.flavour),
            mlx: located.flavour == Flavour::Mlx,
            group: located.affine_group(),
        };
        let (c, d, h, v) = (
            config.sub_channels,
            config.d_model,
            config.pred_hidden,
            config.vocab,
        );
        let freq_out = {
            let f = |x: usize| (x + 3 - 3) / 2 + 1;
            f(f(f(config.mels)))
        };
        let conv0 = l.conv(Stem::Conv0, &[c, 1, 3, 3])?;
        let conv0_bias = f32_param(ops, store, &l.names.bias(Stem::Conv0), c)?.0;
        let stage = |s: usize| -> Result<SubStage, AudioError> {
            Ok(SubStage {
                dw: l.conv(Stem::SubDw(s), &[c, 1, 3, 3])?,
                dw_bias: f32_param(ops, store, &l.names.bias(Stem::SubDw(s)), c)?.0,
                pw: l.lin(Stem::SubPw(s), c, c, true)?,
            })
        };
        let stages = [stage(1)?, stage(2)?];
        let sub_out = l.lin(Stem::SubOut, d, c * freq_out, true)?;

        let max_r = config.lookaheads.iter().copied().max().unwrap_or(0);
        let rel_rows = config.att_left + 2 * max_r + 1;
        let rel_d_min = -i32::try_from(max_r).map_err(|_| checkpoint("lookahead".into()))?;
        let pe: Vec<f32> = (0..rel_rows)
            .flat_map(|r| sinusoid(rel_d_min + i32::try_from(r).unwrap_or(0), d))
            .collect();
        let pe_dev = upload_f32(ops, &pe)?;

        let mut layers = Vec::with_capacity(config.layers);
        for n in 0..config.layers {
            let (bu, bv) = l.names.pos_biases(n);
            let pos = l.lin(Stem::Pos(n), d, d, false)?;
            let rel = ops.scratch(rel_rows * d)?;
            pos.forward(ops, pe_dev.view(), rel.view(), rel_rows)?;
            layers.push(Block {
                norm_ff1: l.norm(Stem::NormFf(n, 1), d)?,
                ff1: (
                    l.lin(Stem::Ff(n, 1, 1), config.ff, d, false)?,
                    l.lin(Stem::Ff(n, 1, 2), d, config.ff, false)?,
                ),
                norm_att: l.norm(Stem::NormAtt(n), d)?,
                q: l.lin(Stem::Q(n), d, d, false)?,
                k: l.lin(Stem::K(n), d, d, false)?,
                v: l.lin(Stem::V(n), d, d, false)?,
                o: l.lin(Stem::O(n), d, d, false)?,
                bias_u: f32_param(ops, store, &bu, d)?.0,
                bias_v: f32_param(ops, store, &bv, d)?.0,
                rel,
                norm_conv: l.norm(Stem::NormConv(n), d)?,
                pw1: l.lin(Stem::Pw1(n), 2 * d, d, false)?,
                dw: l.conv(Stem::Dw(n), &[d, 1, config.conv_kernel])?,
                conv_norm: l.norm(Stem::ConvNorm(n), d)?,
                pw2: l.lin(Stem::Pw2(n), d, d, false)?,
                norm_ff2: l.norm(Stem::NormFf(n, 2), d)?,
                ff2: (
                    l.lin(Stem::Ff(n, 2, 1), config.ff, d, false)?,
                    l.lin(Stem::Ff(n, 2, 2), d, config.ff, false)?,
                ),
                norm_out: l.norm(Stem::NormOut(n), d)?,
            });
        }

        // Prompt linear 1 reads [encoder | one-hot]; the one-hot half is a
        // per-language bias column, so split it off on the host.
        let wide = d + config.prompts;
        let p1 = l.host_matrix(Stem::Prompt1, config.prompt_hidden, wide)?;
        let mut enc_cols = Vec::with_capacity(config.prompt_hidden * d);
        let mut prompt_cols = vec![0.0f32; config.prompts * config.prompt_hidden];
        for (r, row) in p1.chunks_exact(wide).enumerate() {
            enc_cols.extend_from_slice(&row[..d]);
            for (id, value) in row[d..].iter().enumerate() {
                prompt_cols[id * config.prompt_hidden + r] = *value;
            }
        }
        let prompt_w1 = Linear {
            weight: LinearWeight::Gemm {
                dtype: GemmType::F32,
                buffer: upload_f32(ops, &enc_cols)?,
                in_dim: d,
                out_dim: config.prompt_hidden,
            },
            bias: None,
        };
        let prompt_b1 = dequant(&store.require(&l.names.bias(Stem::Prompt1))?)?;
        if prompt_b1.len() != config.prompt_hidden {
            return Err(checkpoint(format!(
                "{} is not {} wide",
                l.names.bias(Stem::Prompt1),
                config.prompt_hidden
            )));
        }

        let embed = upload_f32(ops, &l.host_matrix(Stem::Embed, v, h)?)?;
        let mut lstm = Vec::with_capacity(config.pred_layers);
        for layer in 0..config.pred_layers {
            let (ih, hh) = l.names.lstm_weights(layer);
            let group = l.group;
            let ih_src = LinearSource {
                weight: &ih,
                bias: None,
                affine_group: group,
            };
            let hh_src = LinearSource {
                weight: &hh,
                bias: None,
                affine_group: group,
            };
            let mut w_ih = linear(ops, store, &ih_src, 4 * h, h)?;
            let w_hh = linear(ops, store, &hh_src, 4 * h, h)?;
            let bias = match l.names.lstm_bias(layer) {
                LstmBias::Folded(name) => dequant(&store.require(&name)?)?,
                LstmBias::Separate(ih_name, hh_name) => {
                    let ih_bias = dequant(&store.require(&ih_name)?)?;
                    let hh_bias = dequant(&store.require(&hh_name)?)?;
                    if ih_bias.len() != 4 * h || hh_bias.len() != 4 * h {
                        return Err(checkpoint(format!(
                            "{ih_name} and {hh_name} hold {} and {} values, not {}",
                            ih_bias.len(),
                            hh_bias.len(),
                            4 * h
                        )));
                    }
                    ih_bias.iter().zip(&hh_bias).map(|(x, y)| x + y).collect()
                }
            };
            if bias.len() != 4 * h {
                return Err(checkpoint(format!(
                    "LSTM layer {layer}'s folded bias holds {} values, not {}",
                    bias.len(),
                    4 * h
                )));
            }
            w_ih.bias = Some(upload_f32(ops, &bias)?);
            lstm.push(LstmLayer { w_ih, w_hh });
        }

        Ok(Self {
            conv0,
            conv0_bias,
            stages,
            sub_out,
            layers,
            rel_d_min,
            rel_rows,
            prompt_w1,
            prompt_b1,
            prompt_cols,
            prompt_w2: l.lin(Stem::Prompt2, d, config.prompt_hidden, true)?,
            enc_proj: l.lin(Stem::EncProj, config.joint, d, true)?,
            embed,
            lstm,
            dec_proj: l.lin(Stem::DecProj, config.joint, h, true)?,
            head: l.lin(Stem::JointOut, v, config.joint, true)?,
        })
    }
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
