//! The Gemma 4 audio tower's weights, from the MLX directory or the mmproj GGUF.
//!
//! Both containers hold the same tower. The MLX export keeps it in bf16 beside
//! the language model and quantises only the projection into it; the mmproj
//! GGUF names every tensor differently, stores the per-dimension query scale
//! with its softplus already applied, and swaps the two conv-module norm names
//! against their roles. Those differences are resolved here, once, into
//! [`AudioWeights`].

use std::path::{Path, PathBuf};

use serde_json::Value;
use shiso_audio::AudioError;
use shiso_core::DeviceBuffer;

use crate::audio_ops::load::{LinearSource, conv_weight, f32_param, linear, upload_f32};
use crate::audio_ops::store::{TensorStore, dequant};
use crate::audio_ops::{Linear, Ops};

fn checkpoint(what: impl Into<String>) -> AudioError {
    AudioError::Checkpoint { what: what.into() }
}

/// A linear whose input and output are clamped to recorded bounds.
#[derive(Debug)]
pub(crate) struct Clipped {
    pub(crate) linear: Linear,
    pub(crate) input: (f32, f32),
    pub(crate) output: (f32, f32),
}

/// One conformer layer.
#[derive(Debug)]
pub(crate) struct Layer {
    pub(crate) ff: [FeedForward; 2],
    pub(crate) norm_pre_attn: DeviceBuffer,
    pub(crate) q: Clipped,
    pub(crate) k: Clipped,
    pub(crate) v: Clipped,
    pub(crate) post: Clipped,
    /// `[13][d_model]`: relative-position keys for distances 0..=12.
    pub(crate) rel: DeviceBuffer,
    /// `[head_dim]`: the query's per-dimension scale, softplus and constants applied.
    pub(crate) q_dim_scale: DeviceBuffer,
    pub(crate) norm_post_attn: DeviceBuffer,
    pub(crate) conv_pre_norm: DeviceBuffer,
    pub(crate) conv_start: Clipped,
    /// `[d_model][kernel]`.
    pub(crate) conv_dw: DeviceBuffer,
    pub(crate) conv_norm: DeviceBuffer,
    pub(crate) conv_end: Clipped,
    pub(crate) norm_out: DeviceBuffer,
}

/// A half-step feed-forward.
#[derive(Debug)]
pub(crate) struct FeedForward {
    pub(crate) pre: DeviceBuffer,
    pub(crate) up: Clipped,
    pub(crate) down: Clipped,
    pub(crate) post: DeviceBuffer,
}

/// The tower's dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dims {
    pub(crate) d_model: usize,
    pub(crate) layers: usize,
    pub(crate) heads: usize,
    pub(crate) ff: usize,
    pub(crate) mels: usize,
    pub(crate) channels: [usize; 2],
    pub(crate) kernel: usize,
    pub(crate) out: usize,
    pub(crate) text: usize,
}

/// Every tensor of the tower on the device.
#[derive(Debug)]
pub(crate) struct AudioWeights {
    pub(crate) dims: Dims,
    /// `[C0][1][3][3]`, `[C1][C0][3][3]`.
    pub(crate) conv: [DeviceBuffer; 2],
    pub(crate) conv_norm: [DeviceBuffer; 2],
    pub(crate) input_proj: Linear,
    pub(crate) layers: Vec<Layer>,
    pub(crate) output_proj: Linear,
    pub(crate) embed: Linear,
}

/// The two containers, and what distinguishes their names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Container {
    Mlx,
    Mmproj,
}

/// Relative positions the table covers: the query and the twelve before it.
const REL_ROWS: usize = 13;

struct Loader<'a> {
    ops: &'a Ops,
    store: &'a TensorStore,
    container: Container,
    group: Option<usize>,
}

impl Loader<'_> {
    fn param(&self, name: &str, elements: usize) -> Result<DeviceBuffer, AudioError> {
        Ok(f32_param(self.ops, self.store, name, elements)?.0)
    }

    fn scalar(&self, name: &str) -> Result<f32, AudioError> {
        let values = dequant(&self.store.require(name)?)?;
        match values.as_slice() {
            [v] if v.is_finite() => Ok(*v),
            _ => Err(checkpoint(format!("{name} is not one finite value"))),
        }
    }

    fn linear(
        &self,
        weight: &str,
        bias: Option<&str>,
        out: usize,
        inp: usize,
    ) -> Result<Linear, AudioError> {
        let src = LinearSource {
            weight,
            bias,
            affine_group: self.group,
        };
        linear(self.ops, self.store, &src, out, inp)
    }

    /// A clamped linear at `stem`: `stem.{input,output}_{min,max}` beside the weight.
    fn clipped(&self, stem: &str, out: usize, inp: usize) -> Result<Clipped, AudioError> {
        let weight = match self.container {
            Container::Mlx => format!("{stem}.linear.weight"),
            Container::Mmproj => format!("{stem}.weight"),
        };
        Ok(Clipped {
            linear: self.linear(&weight, None, out, inp)?,
            input: (
                self.scalar(&format!("{stem}.input_min"))?,
                self.scalar(&format!("{stem}.input_max"))?,
            ),
            output: (
                self.scalar(&format!("{stem}.output_min"))?,
                self.scalar(&format!("{stem}.output_max"))?,
            ),
        })
    }
}

/// `[sin(d w) | cos(d w)]` with `w_m = exp(-m ln(1e4) / (half - 1))`.
fn position_row(d: usize, width: usize) -> Vec<f32> {
    let half = width / 2;
    let step = 10_000f64.ln() / crate::audio_ops::f64_of(half.saturating_sub(1).max(1));
    let pos = crate::audio_ops::f64_of(d);
    let mut row = vec![0.0f32; width];
    for m in 0..half {
        let angle = pos * (-crate::audio_ops::f64_of(m) * step).exp();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the table is f32 by contract"
        )]
        {
            row[m] = angle.sin() as f32;
            row[half + m] = angle.cos() as f32;
        }
    }
    row
}

fn softplus(x: f32) -> f32 {
    x.exp().ln_1p()
}

fn usize_at(v: &Value, pointer: &str) -> Result<usize, AudioError> {
    v.pointer(pointer)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| checkpoint(format!("config.json has no {pointer}")))
}

impl AudioWeights {
    /// Loads the tower from an MLX directory or an mmproj GGUF.
    ///
    /// # Errors
    ///
    /// [`AudioError::Checkpoint`] naming the missing or misshapen tensor, or a
    /// container that holds no audio tower; a device error.
    #[expect(
        clippy::too_many_lines,
        reason = "reading two containers' dimensions is one block per container"
    )]
    pub(crate) fn load(ops: &Ops, path: &Path) -> Result<Self, AudioError> {
        let (store, container, dims, group) = if path.is_dir() {
            let config: Value =
                serde_json::from_str(&std::fs::read_to_string(path.join("config.json")).map_err(
                    |source| AudioError::Io {
                        what: "reading config.json",
                        source,
                    },
                )?)
                .map_err(|e| checkpoint(format!("config.json: {e}")))?;
            let index: Value = serde_json::from_str(
                &std::fs::read_to_string(path.join("model.safetensors.index.json")).map_err(
                    |source| AudioError::Io {
                        what: "reading the safetensors index",
                        source,
                    },
                )?,
            )
            .map_err(|e| checkpoint(format!("the safetensors index: {e}")))?;
            let mut files: Vec<String> = index
                .get("weight_map")
                .and_then(Value::as_object)
                .ok_or_else(|| checkpoint("the safetensors index has no weight_map"))?
                .values()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            files.sort();
            files.dedup();
            let paths: Vec<PathBuf> = files.iter().map(|f| path.join(f)).collect();
            let a = config
                .get("audio_config")
                .ok_or_else(|| checkpoint("config.json has no audio_config"))?;
            let channels = a
                .get("subsampling_conv_channels")
                .and_then(Value::as_array)
                .and_then(|c| {
                    Some([
                        usize::try_from(c.first()?.as_u64()?).ok()?,
                        usize::try_from(c.get(1)?.as_u64()?).ok()?,
                    ])
                })
                .ok_or_else(|| {
                    checkpoint("audio_config.subsampling_conv_channels is not two counts")
                })?;
            let d_model = usize_at(a, "/hidden_size")?;
            let dims = Dims {
                d_model,
                layers: usize_at(a, "/num_hidden_layers")?,
                heads: usize_at(a, "/num_attention_heads")?,
                ff: 4 * d_model,
                mels: 128,
                channels,
                kernel: usize_at(a, "/conv_kernel_size")?,
                out: usize_at(a, "/output_proj_dims")?,
                text: usize_at(&config, "/text_config/hidden_size")?,
            };
            let group = config
                .pointer("/quantization/group_size")
                .and_then(Value::as_u64)
                .and_then(|g| usize::try_from(g).ok());
            (
                TensorStore::open_shards(&paths)?,
                Container::Mlx,
                dims,
                group,
            )
        } else {
            let store = TensorStore::open_gguf(path)?;
            let meta = store
                .gguf()
                .ok_or_else(|| checkpoint("not a GGUF"))?
                .metadata();
            if meta.get_str("clip.audio.projector_type") != Some("gemma4a") {
                return Err(checkpoint(format!(
                    "{} holds no Gemma 4 audio tower",
                    path.display()
                )));
            }
            let u = |key: &str| -> Result<usize, AudioError> {
                meta.get_u32(key)
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| checkpoint(format!("the mmproj has no {key}")))
            };
            let d_model = u("clip.audio.embedding_length")?;
            let conv0 = store.require("a.conv1d.0.weight")?;
            let conv1 = store.require("a.conv1d.1.weight")?;
            let out_proj = store.require("a.pre_encode.out.weight")?;
            let dw = store.require("a.blk.0.conv_dw.weight")?;
            let dim = |t: &crate::audio_ops::store::Stored<'_>, i: usize| {
                t.shape
                    .get(i)
                    .and_then(|d| usize::try_from(*d).ok())
                    .unwrap_or(0)
            };
            let dims = Dims {
                d_model,
                layers: u("clip.audio.block_count")?,
                heads: u("clip.audio.attention.head_count")?,
                ff: u("clip.audio.feed_forward_length")?,
                mels: u("clip.audio.num_mel_bins")?,
                channels: [dim(&conv0, 0), dim(&conv1, 0)],
                kernel: dim(&dw, 1),
                out: dim(&out_proj, 0),
                text: u("clip.audio.projection_dim")?,
            };
            (store, Container::Mmproj, dims, None)
        };
        let l = Loader {
            ops,
            store: &store,
            container,
            group,
        };
        load_all(&l, dims).map_err(|e| match e {
            AudioError::Checkpoint { what } => {
                checkpoint(format!("{container:?} audio tower: {what}"))
            }
            other => other,
        })
    }
}

#[expect(
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "one statement per tensor role is the whole loader, and the widths keep short names"
)]
fn load_all(l: &Loader<'_>, dims: Dims) -> Result<AudioWeights, AudioError> {
    let (d, [c0, c1]) = (dims.d_model, dims.channels);
    let head_dim = d / dims.heads.max(1);
    let (mlx, ops) = (l.container == Container::Mlx, l.ops);
    let conv_names = |i: usize| match l.container {
        Container::Mlx => (
            format!("audio_tower.subsample_conv_projection.layer{i}.conv.weight"),
            format!("audio_tower.subsample_conv_projection.layer{i}.norm.weight"),
        ),
        Container::Mmproj => (
            format!("a.conv1d.{i}.weight"),
            format!("a.conv1d.{i}.norm.weight"),
        ),
    };
    let (w0, n0) = conv_names(0);
    let (w1, n1) = conv_names(1);
    let conv = [
        upload_f32(ops, &conv_weight(l.store, &w0, &[c0, 1, 3, 3], mlx)?)?,
        upload_f32(ops, &conv_weight(l.store, &w1, &[c1, c0, 3, 3], mlx)?)?,
    ];
    let conv_norm = [l.param(&n0, c0)?, l.param(&n1, c1)?];
    let freq = dims.mels.div_ceil(2).div_ceil(2);
    let input_proj = match l.container {
        Container::Mlx => l.linear(
            "audio_tower.subsample_conv_projection.input_proj_linear.weight",
            None,
            d,
            c1 * freq,
        )?,
        Container::Mmproj => l.linear("a.input_projection.weight", None, d, c1 * freq)?,
    };
    let q_scale = (1.0 / crate::audio_ops::f64_of(head_dim).sqrt()) / std::f64::consts::LN_2;
    let mut layers = Vec::with_capacity(dims.layers);
    for n in 0..dims.layers {
        // Role -> name, per container.
        let (ff_stem, attn, lconv, norm) = match l.container {
            Container::Mlx => {
                let p = format!("audio_tower.layers.{n}.");
                (
                    p.clone(),
                    format!("{p}self_attn."),
                    format!("{p}lconv1d."),
                    p,
                )
            }
            Container::Mmproj => {
                let p = format!("a.blk.{n}.");
                (p.clone(), p.clone(), p.clone(), p)
            }
        };
        let ff = |i: usize| -> Result<FeedForward, AudioError> {
            match l.container {
                Container::Mlx => {
                    let s = format!("{ff_stem}feed_forward{i}.");
                    Ok(FeedForward {
                        pre: l.param(&format!("{s}pre_layer_norm.weight"), d)?,
                        up: l.clipped(&format!("{s}ffw_layer_1"), dims.ff, d)?,
                        down: l.clipped(&format!("{s}ffw_layer_2"), d, dims.ff)?,
                        post: l.param(&format!("{s}post_layer_norm.weight"), d)?,
                    })
                }
                Container::Mmproj => {
                    let sfx = if i == 1 { "" } else { "_1" };
                    Ok(FeedForward {
                        pre: l.param(&format!("{ff_stem}ffn_norm{sfx}.weight"), d)?,
                        up: l.clipped(&format!("{ff_stem}ffn_up{sfx}"), dims.ff, d)?,
                        down: l.clipped(&format!("{ff_stem}ffn_down{sfx}"), d, dims.ff)?,
                        post: l.param(&format!("{ff_stem}ffn_post_norm{sfx}.weight"), d)?,
                    })
                }
            }
        };
        let (q, k, v, post, rel_name, pds_name) = match l.container {
            Container::Mlx => (
                l.clipped(&format!("{attn}q_proj"), d, d)?,
                l.clipped(&format!("{attn}k_proj"), d, d)?,
                l.clipped(&format!("{attn}v_proj"), d, d)?,
                l.clipped(&format!("{attn}post"), d, d)?,
                format!("{attn}relative_k_proj.weight"),
                format!("{attn}per_dim_scale"),
            ),
            Container::Mmproj => (
                l.clipped(&format!("{attn}attn_q"), d, d)?,
                l.clipped(&format!("{attn}attn_k"), d, d)?,
                l.clipped(&format!("{attn}attn_v"), d, d)?,
                l.clipped(&format!("{attn}attn_out"), d, d)?,
                format!("{attn}attn_k_rel.weight"),
                format!("{attn}per_dim_scale.weight"),
            ),
        };
        // The relative-position keys depend only on the weights: one table per layer.
        let rel_linear = l.linear(&rel_name, None, d, d)?;
        let pe: Vec<f32> = (0..REL_ROWS).flat_map(|r| position_row(r, d)).collect();
        let pe_dev = upload_f32(ops, &pe)?;
        let rel = ops.scratch(REL_ROWS * d)?;
        rel_linear.forward(ops, pe_dev.view(), rel.view(), REL_ROWS)?;
        let pds = dequant(&l.store.require(&pds_name)?)?;
        if pds.len() != head_dim {
            return Err(checkpoint(format!(
                "{pds_name} holds {} values, not {head_dim}",
                pds.len()
            )));
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the kernel takes f32 scales"
        )]
        let q_dim: Vec<f32> = pds
            .iter()
            .map(|&p| (q_scale * f64::from(if mlx { softplus(p) } else { p })) as f32)
            .collect();
        let (pre_norm_name, conv_norm_name, start, end, dw_name) = match l.container {
            Container::Mlx => (
                format!("{lconv}pre_layer_norm.weight"),
                format!("{lconv}conv_norm.weight"),
                format!("{lconv}linear_start"),
                format!("{lconv}linear_end"),
                format!("{lconv}depthwise_conv1d.weight"),
            ),
            // The mmproj's two conv norms are named against their roles.
            Container::Mmproj => (
                format!("{lconv}conv_norm.weight"),
                format!("{lconv}norm_conv.weight"),
                format!("{lconv}conv_pw1"),
                format!("{lconv}conv_pw2"),
                format!("{lconv}conv_dw.weight"),
            ),
        };
        let (pre_attn, post_attn, out_norm) = match l.container {
            Container::Mlx => (
                format!("{norm}norm_pre_attn.weight"),
                format!("{norm}norm_post_attn.weight"),
                format!("{norm}norm_out.weight"),
            ),
            Container::Mmproj => (
                format!("{norm}attn_pre_norm.weight"),
                format!("{norm}attn_post_norm.weight"),
                format!("{norm}ln2.weight"),
            ),
        };
        layers.push(Layer {
            ff: [ff(1)?, ff(2)?],
            norm_pre_attn: l.param(&pre_attn, d)?,
            q,
            k,
            v,
            post,
            rel,
            q_dim_scale: upload_f32(ops, &q_dim)?,
            norm_post_attn: l.param(&post_attn, d)?,
            conv_pre_norm: l.param(&pre_norm_name, d)?,
            conv_start: l.clipped(&start, 2 * d, d)?,
            conv_dw: upload_f32(
                ops,
                &conv_weight(l.store, &dw_name, &[d, 1, dims.kernel], mlx)?,
            )?,
            conv_norm: l.param(&conv_norm_name, d)?,
            conv_end: l.clipped(&end, d, d)?,
            norm_out: l.param(&out_norm, d)?,
        });
    }
    let (out_w, out_b, embed_w) = match l.container {
        Container::Mlx => (
            "audio_tower.output_proj.weight",
            "audio_tower.output_proj.bias",
            "embed_audio.embedding_projection.weight",
        ),
        Container::Mmproj => (
            "a.pre_encode.out.weight",
            "a.pre_encode.out.bias",
            "mm.a.input_projection.weight",
        ),
    };
    Ok(AudioWeights {
        dims,
        conv,
        conv_norm,
        input_proj,
        layers,
        output_proj: l.linear(out_w, Some(out_b), dims.out, d)?,
        embed: l.linear(embed_w, None, dims.text, dims.out)?,
    })
}
