//! The block drafter's GGUF container: the same five layers, input
//! projection and selector as [`crate::qwen35::block_draft`]'s directory, in
//! one file whose projections are already block-quantized.
//!
//! Read as stored. The projections stay in the blocks the file wrote them in
//! (`q4_K` and `q6_K` here, both formats the dense matvecs read), the norms
//! and convolution kernels are the file's own `f32`, and the selector's two
//! codebooks stay `q4_K` on the host, one 256-wide super-block a vocabulary
//! row, decoded a row at a time for the candidates a drafted position scores.
//!
//! Names and their directory twins, checked value for value against the
//! `bfloat16` export of the same drafter (every norm and kernel equal):
//!
//! | GGUF | directory |
//! | --- | --- |
//! | `fc.weight` | `fc.weight` |
//! | `enc.output_norm.weight` | `hidden_norm.weight` |
//! | `output_norm.weight` | `norm.weight` |
//! | `selector_hidden.weight` | `candidate_selector.hidden_projection.weight` |
//! | `selector_{predecessor,successor}.weight` | `candidate_selector.{..}_codebook` |
//! | `blk.i.attn_norm` / `ffn_norm` | `layers.i.input_layernorm` / `post_attention_layernorm` |
//! | `blk.i.{attn,ffn}_conv_base` | `layers.i.{attention,mlp}_conv.base_kernel` |
//! | `blk.i.{attn,ffn}_conv_proj.weight` | `layers.i.{attention,mlp}_conv.kernel_projection.weight` |
//!
//! The file's `dflash.target_layers` count hidden states from the embedding
//! (state 0 is the embedding, state `n` the output of layer `n - 1`); the
//! directory's `target_layer_ids` count layers. The taps here are the latter.

use std::path::Path;
use std::sync::Arc;

use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, import_host};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::TokenId;
use shiso_engine::attn::Attn;
use shiso_engine::block_draft::{
    BlockDraftLayer, BlockDraftParts, BlockDraftShape, Codebook, CodebookRows, DynConv,
};
use shiso_engine::ffn::Ffn;
use shiso_engine::projection::Weight;
use shiso_modelfile::metadata::GgufMetadata;
use shiso_modelfile::{ModelFile, TensorDesc};

use crate::qwen35::block_draft::GGUF_ARCHITECTURE;

/// Projection formats this container is read in.
const PROJECTION_TYPES: [DType; 4] = [DType::Q4_K, DType::Q6_K, DType::Q8_0, DType::F16];

/// A file error, as the engine's error type.
#[expect(
    clippy::needless_pass_by_value,
    reason = "a `map_err` adaptor, which is handed the error by value"
)]
fn open_failure(err: shiso_modelfile::GgufError) -> CoreError {
    CoreError::InvalidValue {
        key: "block drafter checkpoint",
        value: err.to_string(),
    }
}

/// A required `dflash.*` number.
fn number(meta: &GgufMetadata, suffix: &'static str) -> CoreResult<u32> {
    meta.get_u32(&format!("{GGUF_ARCHITECTURE}.{suffix}"))
        .ok_or_else(|| CoreError::ConfigMissing {
            key: format!("{GGUF_ARCHITECTURE}.{suffix}"),
        })
}

/// A required `dflash.*` float.
fn float(meta: &GgufMetadata, suffix: &'static str) -> CoreResult<f32> {
    meta.get_f32(&format!("{GGUF_ARCHITECTURE}.{suffix}"))
        .ok_or_else(|| CoreError::ConfigMissing {
            key: format!("{GGUF_ARCHITECTURE}.{suffix}"),
        })
}

/// The open file and its tensor table.
struct File {
    file: ModelFile,
}

impl File {
    fn desc(&self, name: &str) -> CoreResult<&TensorDesc> {
        let ModelFile::Gguf(gguf) = &self.file else {
            return Err(CoreError::InvalidValue {
                key: "block drafter checkpoint",
                value: "not a GGUF file".into(),
            });
        };
        gguf.find(name).ok_or_else(|| CoreError::InvalidValue {
            key: "block drafter tensor",
            value: format!("{name} is not in the drafter's GGUF file"),
        })
    }

    /// A `[out, in]` projection, imported as stored.
    fn linear(&self, backend: &Arc<dyn Backend>, name: &str) -> CoreResult<Weight> {
        let desc = self.desc(name)?;
        if !PROJECTION_TYPES.contains(&desc.dtype) {
            return Err(CoreError::InvalidValue {
                key: "block drafter projection",
                value: format!("{name} is {:?}, which no dense matvec reads", desc.dtype),
            });
        }
        let dims = desc.shape.dims();
        let [cols, rows] = dims else {
            return Err(CoreError::InvalidValue {
                key: "block drafter projection",
                value: format!("{name} has shape {dims:?}, where a projection is two-dimensional"),
            });
        };
        let dim = |d: u64| {
            u32::try_from(d).map_err(|_| CoreError::ShapeMismatch {
                field: "block drafter projection extent",
                expected: u64::from(u32::MAX),
                got: d,
            })
        };
        let (host, range) = self.file.tensor_host(desc).map_err(open_failure)?;
        Weight::import(backend, desc.dtype, dim(*cols)?, dim(*rows)?, host, range)
    }

    /// A norm weight or a convolution kernel, as the file's `f32`.
    fn vector(&self, backend: &Arc<dyn Backend>, name: &str) -> CoreResult<DeviceBuffer> {
        let desc = self.desc(name)?;
        if desc.dtype != DType::F32 {
            return Err(CoreError::InvalidValue {
                key: "block drafter vector",
                value: format!(
                    "{name} is {:?}, where the drafter's vectors are f32",
                    desc.dtype
                ),
            });
        }
        let (host, range) = self.file.tensor_host(desc).map_err(open_failure)?;
        import_host(backend, host, range)
    }

    /// Drafter layer `i`.
    fn layer(&self, backend: &Arc<dyn Backend>, i: u32) -> CoreResult<BlockDraftLayer> {
        let file = self;
        let name = |s: &str| format!("blk.{i}.{s}");
        Ok(BlockDraftLayer {
            input_norm: file.vector(backend, &name("attn_norm.weight"))?,
            post_norm: file.vector(backend, &name("ffn_norm.weight"))?,
            attn: Attn {
                q: file.linear(backend, &name("attn_q.weight"))?,
                k: Some(file.linear(backend, &name("attn_k.weight"))?),
                v: Some(file.linear(backend, &name("attn_v.weight"))?),
                output: file.linear(backend, &name("attn_output.weight"))?,
                q_norm: Some(file.vector(backend, &name("attn_q_norm.weight"))?),
                k_norm: Some(file.vector(backend, &name("attn_k_norm.weight"))?),
            },
            ffn: Ffn {
                gate: Some(file.linear(backend, &name("ffn_gate.weight"))?),
                up: file.linear(backend, &name("ffn_up.weight"))?,
                down: file.linear(backend, &name("ffn_down.weight"))?,
            },
            attn_conv: DynConv {
                base: file.vector(backend, &name("attn_conv_base"))?,
                projection: file.linear(backend, &name("attn_conv_proj.weight"))?,
            },
            mlp_conv: DynConv {
                base: file.vector(backend, &name("ffn_conv_base"))?,
                projection: file.linear(backend, &name("ffn_conv_proj.weight"))?,
            },
        })
    }

    /// A selector codebook: `q4_K`, one super-block a vocabulary row.
    fn codebook(&self, name: &str, rank: u32, vocab: u32) -> CoreResult<Codebook> {
        let desc = self.desc(name)?;
        let dims = desc.shape.dims();
        if desc.dtype != DType::Q4_K || dims != [u64::from(rank), u64::from(vocab)] || rank != 256 {
            return Err(CoreError::InvalidValue {
                key: "block drafter codebook",
                value: format!(
                    "{name} is {:?} {dims:?}, where the selector reads one 256-wide q4_K block \
                     per vocabulary row ({vocab} rows of rank {rank})",
                    desc.dtype
                ),
            });
        }
        let bytes = self.file.tensor_bytes(name).map_err(open_failure)?;
        Ok(Codebook {
            rank,
            rows: CodebookRows::Q4K(bytes.into_owned()),
        })
    }
}

/// Reads the drafter's GGUF file at `path` and puts it on the device.
///
/// # Errors
///
/// As [`crate::qwen35::block_draft::load`]: [`CoreError::InvalidValue`] for a
/// file that is not this drafter or a tensor it does not carry,
/// [`CoreError::ConfigMissing`] for a missing key, [`CoreError::ShapeMismatch`]
/// for one built for another target.
pub(super) fn load(
    backend: &Arc<dyn Backend>,
    path: &Path,
    target: (u32, u32, u32),
) -> CoreResult<BlockDraftParts> {
    let (target_hidden, target_layers, target_vocab) = target;
    let file = File {
        file: ModelFile::open(path).map_err(open_failure)?,
    };
    let ModelFile::Gguf(gguf) = &file.file else {
        return Err(CoreError::InvalidValue {
            key: "block drafter checkpoint",
            value: format!("{} is not a GGUF file", path.display()),
        });
    };
    let meta = gguf.metadata();
    if meta.architecture() != Some(GGUF_ARCHITECTURE) {
        return Err(CoreError::UnknownArch {
            model_type: meta.architecture().unwrap_or("").to_owned(),
        });
    }
    let taps = meta
        .get_array(&format!("{GGUF_ARCHITECTURE}.target_layers"))
        .ok_or_else(|| CoreError::ConfigMissing {
            key: format!("{GGUF_ARCHITECTURE}.target_layers"),
        })?
        .iter()
        .map(|v| match v.map_err(open_failure)? {
            shiso_modelfile::MetaValue::I32(n) if n >= 1 => Ok(n.cast_unsigned() - 1),
            shiso_modelfile::MetaValue::U32(n) if n >= 1 => Ok(n - 1),
            other => Err(CoreError::InvalidValue {
                key: "block drafter target_layers",
                value: format!("{other:?} is not a hidden state past the embedding"),
            }),
        })
        .collect::<CoreResult<Vec<u32>>>()?;
    let mask = meta
        .get_u32("tokenizer.ggml.mask_token_id")
        .ok_or_else(|| CoreError::ConfigMissing {
            key: "tokenizer.ggml.mask_token_id".into(),
        })?;
    let shape = BlockDraftShape {
        hidden: number(meta, "embedding_length")?,
        n_head: number(meta, "attention.head_count")?,
        n_head_kv: number(meta, "attention.head_count_kv")?,
        head_dim: number(meta, "attention.key_length")?,
        intermediate: number(meta, "feed_forward_length")?,
        block: number(meta, "block_size")?,
        window: number(meta, "attention.sliding_window")?,
        conv_taps: number(meta, "conv_kernel_size")?,
        conv_group: number(meta, "conv_group_size")?,
        rope_theta: float(meta, "rope.freq_base")?,
        eps: float(meta, "attention.layer_norm_rms_epsilon")?,
        taps,
        mask_token: TokenId::new(mask),
        selector_rank: number(meta, "selector_rank")?,
        selector_top_k: number(meta, "selector_top_k")?,
    };
    shape.check(target_hidden, target_layers)?;
    let n_layers = number(meta, "block_count")?;
    let clock = std::time::Instant::now();
    let layers = (0..n_layers)
        .map(|i| file.layer(backend, i))
        .collect::<CoreResult<Vec<_>>>()?;
    let rank = shape.selector_rank;
    let parts = BlockDraftParts {
        fc: file.linear(backend, "fc.weight")?,
        hidden_norm: file.vector(backend, "enc.output_norm.weight")?,
        layers,
        norm: file.vector(backend, "output_norm.weight")?,
        hidden_projection: file.linear(backend, "selector_hidden.weight")?,
        predecessor: file.codebook("selector_predecessor.weight", rank, target_vocab)?,
        successor: file.codebook("selector_successor.weight", rank, target_vocab)?,
        shape,
    };
    tracing::info!(
        seconds = clock.elapsed().as_secs_f64(),
        path = %path.display(),
        "block drafter (GGUF) uploaded"
    );
    Ok(parts)
}
