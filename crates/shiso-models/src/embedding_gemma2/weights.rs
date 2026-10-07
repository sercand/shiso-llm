//! The encoder's tensors, by the names the GGUF container gives them.
//!
//! Every matrix the pass multiplies stays in the file's own encoding - `BF16`
//! or `Q8_0` - and is imported from the mapping as it is; the audio set's GEMM
//! reads both. Three things are deliberately NOT on the device:
//!
//! - **The token table.** 262,144 rows of which an input reads a few hundred.
//!   Its rows are decoded on the host straight out of the mapping, which keeps
//!   134 to 268 MB off the device for a model whose whole transformer is a
//!   fraction of that.
//! - **The output projection.** It is linear and has no bias, so projecting the
//!   mean of the final rows is projecting every row and then taking the mean -
//!   one 768x512 product per input instead of one per token, on the host.
//! - **The layer scalars.** One float each, folded into a launch's argument.

use std::sync::Arc;

use shiso_audio::AudioError;
use shiso_core::{DeviceBuffer, HostBytes};

use super::config::EncoderConfig;
use crate::audio_ops::load::{LinearSource, f32_param, linear, linear_rows};
use crate::audio_ops::store::{StoredType, TensorStore, dequant};
use crate::audio_ops::{Linear, Ops};
use crate::embedder::EmbedError;

/// Bytes of one `Q8_0` block: an f16 scale and 32 signed bytes.
const Q8_0_BLOCK_BYTES: usize = 34;
/// Elements one `Q8_0` block holds.
const Q8_0_BLOCK: usize = 32;

/// One layer's resident tensors.
#[derive(Debug)]
pub struct LayerWeights {
    /// Pre-attention norm, `[hidden]`.
    pub attn_norm: DeviceBuffer,
    /// Query projection, `hidden -> q_heads * head_dim`.
    pub q: Linear,
    /// Key projection, `hidden -> kv_heads * head_dim`.
    pub k: Linear,
    /// Value projection, as the key.
    pub v: Linear,
    /// Per-head query norm, `[head_dim]`.
    pub q_norm: DeviceBuffer,
    /// Per-head key norm, `[head_dim]`.
    pub k_norm: DeviceBuffer,
    /// Output projection, `q_heads * head_dim -> hidden`.
    pub o: Linear,
    /// Norm on the attention block's output, `[hidden]`.
    pub post_attn_norm: DeviceBuffer,
    /// Pre-feed-forward norm, `[hidden]`.
    pub ffn_norm: DeviceBuffer,
    /// Gate projection, `hidden -> intermediate`.
    pub gate: Linear,
    /// Up projection, `hidden -> intermediate`.
    pub up: Linear,
    /// Down projection, `intermediate -> hidden`.
    pub down: Linear,
    /// Norm on the feed-forward block's output, `[hidden]`.
    pub post_ffn_norm: DeviceBuffer,
    /// The per-layer block's gate, `hidden -> ple_dim`.
    pub ple_gate: Linear,
    /// The per-layer block's projection back, `ple_dim -> hidden`.
    pub ple_proj: Linear,
    /// Norm on the per-layer block's output, `[hidden]`.
    pub ple_norm: DeviceBuffer,
    /// The scale the layer's output is multiplied by.
    pub scalar: f32,
}

/// The token table, read on the host.
#[derive(Debug)]
pub struct TokenTable {
    host: Arc<dyn HostBytes>,
    start: usize,
    encoding: RowEncoding,
    width: usize,
    rows: usize,
}

/// How one table row is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowEncoding {
    F32,
    F16,
    Bf16,
    Q8_0,
}

impl RowEncoding {
    fn row_bytes(self, width: usize) -> usize {
        match self {
            Self::F32 => width * 4,
            Self::F16 | Self::Bf16 => width * 2,
            Self::Q8_0 => width / Q8_0_BLOCK * Q8_0_BLOCK_BYTES,
        }
    }
}

impl TokenTable {
    /// Writes row `id`, decoded, into `out`.
    ///
    /// # Errors
    ///
    /// [`EmbedError::Input`] for an id past the table: a token-id input is the
    /// caller's, and an id the vocabulary does not have is the caller's
    /// mistake.
    pub fn row(&self, id: usize, out: &mut [f32]) -> Result<(), EmbedError> {
        if id >= self.rows {
            return Err(EmbedError::Input {
                what: format!("token id {id} is past this model's {} tokens", self.rows),
            });
        }
        let len = self.encoding.row_bytes(self.width);
        let at = self.start + id * len;
        let bytes = self
            .host
            .as_ref()
            .as_ref()
            .get(at..at + len)
            .ok_or_else(|| EmbedError::Checkpoint {
                what: format!("token row {id} runs past the mapping"),
            })?;
        match self.encoding {
            RowEncoding::F32 => {
                for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                    *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            RowEncoding::F16 => {
                for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *o = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
            RowEncoding::Bf16 => {
                for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *o = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
            RowEncoding::Q8_0 => {
                for (o, block) in out
                    .chunks_exact_mut(Q8_0_BLOCK)
                    .zip(bytes.chunks_exact(Q8_0_BLOCK_BYTES))
                {
                    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                    for (o, q) in o.iter_mut().zip(&block[2..]) {
                        *o = scale * f32::from(i8::from_le_bytes([*q]));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Everything the encoder reads.
#[derive(Debug)]
pub struct EncoderWeights {
    /// The token table.
    pub tokens: TokenTable,
    /// Projects an input row to one layer's per-layer slice,
    /// `hidden -> ple_dim`, one per layer.
    ///
    /// The checkpoint stores one fused `[n_layers * ple_dim][hidden]` matrix.
    /// It is imported as one row range per layer so that a pass holds one
    /// layer's slice at a time: the fused product at the longest input is
    /// 8192 x 12288 floats, 400 MB of scratch, for values each layer reads
    /// once.
    pub ple_model_proj: Vec<Linear>,
    /// Norm on each slice, `[ple_dim]`.
    pub ple_proj_norm: DeviceBuffer,
    /// The layers, in order.
    pub layers: Vec<LayerWeights>,
    /// The final norm, `[hidden]`.
    pub output_norm: DeviceBuffer,
    /// The output projection, `[out_dim][hidden]`, on the host.
    pub output: Vec<f32>,
}

impl EncoderWeights {
    /// Imports every tensor `config` says the encoder has.
    ///
    /// # Errors
    ///
    /// [`EmbedError::Checkpoint`] naming the first tensor that is absent,
    /// misshapen or stored in an encoding nothing here reads.
    pub fn load(
        ops: &Ops,
        store: &TensorStore,
        config: &EncoderConfig,
    ) -> Result<Self, EmbedError> {
        let h = config.hidden;
        let layers = config
            .layers
            .iter()
            .enumerate()
            .map(|(i, dims)| {
                let q = config.q_heads * dims.head_dim;
                let kv = dims.kv_heads * dims.head_dim;
                let name = |suffix: &str| format!("blk.{i}.{suffix}.weight");
                let lin = |suffix: &str, out: usize, inp: usize| {
                    let weight = name(suffix);
                    linear(
                        ops,
                        store,
                        &LinearSource {
                            weight: &weight,
                            bias: None,
                            affine_group: None,
                        },
                        out,
                        inp,
                    )
                };
                let vector = |suffix: &str, n: usize| -> Result<DeviceBuffer, AudioError> {
                    Ok(f32_param(ops, store, &name(suffix), n)?.0)
                };
                let scalar = f32_param(ops, store, &name("layer_output_scale"), 1)?.1[0];
                Ok::<_, AudioError>(LayerWeights {
                    attn_norm: vector("attn_norm", h)?,
                    q: lin("attn_q", q, h)?,
                    k: lin("attn_k", kv, h)?,
                    v: lin("attn_v", kv, h)?,
                    q_norm: vector("attn_q_norm", dims.head_dim)?,
                    k_norm: vector("attn_k_norm", dims.head_dim)?,
                    o: lin("attn_output", h, q)?,
                    post_attn_norm: vector("post_attention_norm", h)?,
                    ffn_norm: vector("ffn_norm", h)?,
                    gate: lin("ffn_gate", config.intermediate, h)?,
                    up: lin("ffn_up", config.intermediate, h)?,
                    down: lin("ffn_down", h, config.intermediate)?,
                    post_ffn_norm: vector("post_ffw_norm", h)?,
                    ple_gate: lin("inp_gate", config.ple_dim, h)?,
                    ple_proj: lin("proj", h, config.ple_dim)?,
                    ple_norm: vector("post_norm", h)?,
                    scalar,
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(checkpoint)?;
        let fused = config.n_layers() * config.ple_dim;
        let ple_model_proj = (0..config.n_layers())
            .map(|l| {
                let rows = l * config.ple_dim..(l + 1) * config.ple_dim;
                linear_rows(ops, store, "per_layer_model_proj.weight", rows, fused, h)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(checkpoint)?;
        let ple_proj_norm = f32_param(ops, store, "per_layer_proj_norm.weight", config.ple_dim)
            .map_err(checkpoint)?
            .0;
        let output_norm = f32_param(ops, store, "output_norm.weight", h)
            .map_err(checkpoint)?
            .0;
        let output = store.require("output.weight").map_err(checkpoint)?;
        if output.shape != [config.out_dim as u64, h as u64] {
            return Err(EmbedError::Checkpoint {
                what: format!(
                    "output.weight has shape {:?}, not [{}][{h}]",
                    output.shape, config.out_dim
                ),
            });
        }
        let output = dequant(&output).map_err(checkpoint)?;
        Ok(Self {
            tokens: token_table(store, config)?,
            ple_model_proj,
            ple_proj_norm,
            layers,
            output_norm,
            output,
        })
    }
}

fn token_table(store: &TensorStore, config: &EncoderConfig) -> Result<TokenTable, EmbedError> {
    let name = "token_embd.weight";
    let t = store.require(name).map_err(checkpoint)?;
    let encoding = match t.ty {
        StoredType::F32 => RowEncoding::F32,
        StoredType::F16 => RowEncoding::F16,
        StoredType::Bf16 => RowEncoding::Bf16,
        StoredType::Q8_0 if config.hidden.is_multiple_of(Q8_0_BLOCK) => RowEncoding::Q8_0,
        ref other => {
            return Err(EmbedError::Checkpoint {
                what: format!("{name} is stored as {other:?}, which no row decoder here reads"),
            });
        }
    };
    if t.shape != [config.vocab as u64, config.hidden as u64] {
        return Err(EmbedError::Checkpoint {
            what: format!(
                "{name} has shape {:?}, not [{}][{}]",
                t.shape, config.vocab, config.hidden
            ),
        });
    }
    if t.range.len() < encoding.row_bytes(config.hidden) * config.vocab {
        return Err(EmbedError::Checkpoint {
            what: format!("{name} is shorter than its shape"),
        });
    }
    Ok(TokenTable {
        host: Arc::clone(&t.host),
        start: t.range.start,
        encoding,
        width: config.hidden,
        rows: config.vocab,
    })
}

fn checkpoint(error: AudioError) -> EmbedError {
    match error {
        AudioError::Checkpoint { what } => EmbedError::Checkpoint { what },
        other => EmbedError::Audio(other),
    }
}
