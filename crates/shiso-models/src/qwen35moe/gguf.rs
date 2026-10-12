//! This family's weights, read out of a GGUF container.
//!
//! The sibling of `super::model`'s `import_weights`, which reads the same
//! roles out of a `modelopt` safetensors export. Both end at one
//! [`shiso_engine::layers::Weights`] and the stack cannot tell them apart.
//!
//! # Why this exists
//!
//! The NVFP4 export this family was ported against is a **bad quantization**:
//! its body collapses onto one CJK token, established in `transformers` by
//! `scripts/qwen35moe_mtp_reference.py` with no part of this tree involved, and
//! controlled for by llama.cpp's fluent output from a GGUF of the same base
//! model. This reader is what gives the architecture a container that answers.
//!
//! # The three things a GGUF does differently, and they are all load-bearing
//!
//! **The value heads are already in this engine's order.** The `modelopt`
//! reader permutes eight tensors per recurrent slot, because `transformers`
//! builds the query and key heads with `repeat_interleave` - value head
//! `k * group + m` reads key head `k` - where every kernel here reads key head
//! `h % key_heads`. Nothing below permutes anything, and that is not an
//! omission: llama.cpp's converter has already done it, which is why
//! `crates/shiso-models/src/qwen35/model.rs` - the dense family's GGUF reader -
//! does not permute either while its `modelopt` sibling does. A reader that
//! permuted here would undo the converter's work and the model would run,
//! stay finite through all forty slots, and answer fluent nonsense.
//!
//! **The routed experts arrive stacked.** `modelopt` stores 256 separate
//! tensors per projection per slot and the loader stacks them; a GGUF stores
//! one `[in_dim, out_dim, experts]` tensor, which is already the
//! `[expert][row][element]` memory order the expert entries stride through. So
//! a slab here is one allocation and one copy rather than 256 of each.
//!
//! **A routed block holds two codecs.** `UD-Q4_K_XL` keeps gate and up in
//! `Q4_K` and down in `Q5_K`, which is what made `ExpertCodec` a property of a
//! stage rather than of a block.
//!
//! # What is not read here
//!
//! `blk.<n_layers>` and anything past it. Those are the `NextN` blocks - the
//! multi-token-prediction head - and `nextn_predict_layers` has already been
//! subtracted from the slot count by [`super::config_from_gguf`]. The head is a
//! different function over the same residual width, so a reader that ran it as
//! a forty-first layer would find every tensor it looked for.

use std::path::Path;
use std::sync::Arc;

use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, import_host};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::LayerIdx;
use shiso_engine::attn::Attn;
use shiso_engine::ffn::Ffn;
use shiso_engine::gdn::Gdn;
use shiso_engine::layers::{Block, LayerWeights, Routed, Weights};
use shiso_engine::moe::{ActDtype, ExpertCodec, ExpertSlab, Moe};
use shiso_engine::projection::Weight;
use shiso_engine::shape::{HybridShape, LayerKind};
use shiso_modelfile::reader::GgufFile;
use shiso_modelfile::{GgufError, ModelFile, TensorDesc};

use crate::qwen35moe::config::Qwen35MoeConfig;
use crate::qwen35moe::gguf_config::config_from_gguf;

/// The one tensor whose row count is this family's vocabulary.
const EMBED: &str = "token_embd.weight";

/// Formats this reader accepts for a projection.
///
/// The container's own set plus the two it does not use, so that a re-quantized
/// file of the same architecture loads without an edit here. Anything else
/// means the file is not what it claims; the engine refuses separately, and
/// with a different message, a format for which this build has no matvec.
const WEIGHT_TYPES: &[DType] = &[
    DType::Q4_0,
    DType::Q4_K,
    DType::Q5_K,
    DType::Q6_K,
    DType::Q8_0,
    DType::F16,
    DType::F32,
];

/// Opens a GGUF of this architecture and reads its geometry.
///
/// Separate from the weight import so that a caller can learn what a file is
/// before paying for a device.
///
/// # Errors
///
/// Whatever the container reports, [`CoreError::UnknownArch`] when the file
/// declares another architecture, and [`CoreError::ConfigMissing`] naming a key
/// this family cannot default.
pub fn open_gguf(path: &Path) -> CoreResult<(ModelFile, Qwen35MoeConfig)> {
    let file = ModelFile::open(path).map_err(open_failure)?;
    let vocab = {
        let desc = find(&file, EMBED)?;
        // `ne` order: the embedding is `[hidden, vocab]`, so the vocabulary is
        // the SECOND dimension. Reading the first would give a 2,048-token
        // vocabulary and a logits row that is 1% of the one the sampler wants.
        extent(desc.shape.dims().get(1).copied().unwrap_or(0))?
    };
    let config = config_from_gguf(inner(&file)?.metadata(), vocab)?;
    Ok((file, config))
}

/// The GGUF behind a container, or the refusal for a directory.
///
/// `ModelFile` opens a directory as a sharded safetensors export, which this
/// architecture has a reader for already - `super::model::load`. A caller that
/// reached here with one asked for the GGUF path by name, so it is a typed
/// refusal rather than a silent fall back to the other reader.
fn inner(file: &ModelFile) -> CoreResult<&GgufFile> {
    match file {
        ModelFile::Gguf(gguf) => Ok(gguf),
        // Every other container, including any this enum gains later. A
        // reader that fell through to one of those would be guessing at a
        // layout rather than refusing an unhandled one.
        _ => Err(CoreError::Unsupported {
            what: "a container that is not a gguf handed to the gguf reader",
            backend: "qwen35moe",
        }),
    }
}

/// Moves every tensor of a GGUF onto the device, in the order the pass reads
/// them.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] naming the tensor when one is absent,
/// [`CoreError::Unsupported`] when one is stored in a format this reader does
/// not accept, and whatever the device reports.
pub fn import_weights_gguf(
    backend: &Arc<dyn Backend>,
    file: &ModelFile,
    shape: &HybridShape,
) -> CoreResult<Weights> {
    let mut gdn_rank = 0u32;
    let mut attn_rank = 0u32;
    let mut layers = Vec::with_capacity(shape.layers.len());
    for (slot, kind) in shape.layers.iter().enumerate() {
        let il = LayerIdx::new(u32::try_from(slot).unwrap_or(u32::MAX));
        let kv_rank = shape.kv_owner_rank(il);
        let (block, ordinal) = match kind {
            LayerKind::Gdn => {
                let rank = gdn_rank;
                gdn_rank += 1;
                (Block::Gdn(import_gdn(backend, file, slot)?), rank)
            }
            LayerKind::Attn => {
                let rank = attn_rank;
                attn_rank += 1;
                (Block::Attn(import_attn(backend, file, slot)?), rank)
            }
            // This family's shape holds neither; its config never states one.
            LayerKind::Ssm | LayerKind::ShortConv => {
                return Err(CoreError::Unsupported {
                    what: "a state-space or short-convolution slot in this family",
                    backend: "qwen35",
                });
            }
        };
        layers.push(LayerWeights {
            input_norm: vector(backend, file, &name(slot, "attn_norm.weight"))?,
            // The pre-only scheme this family has: nothing normalizes a block's
            // output and the residual sum carries no scalar.
            post_attn_norm: None,
            ffn_norm: Some(vector(
                backend,
                file,
                &name(slot, "post_attention_norm.weight"),
            )?),
            post_ffn_norm: None,
            // The SHARED expert takes the dense feed-forward's own field,
            // because that is what it is; the routed half is beside it.
            ffn: Some(Ffn {
                gate: Some(weight(backend, file, &name(slot, "ffn_gate_shexp.weight"))?),
                up: weight(backend, file, &name(slot, "ffn_up_shexp.weight"))?,
                down: weight(backend, file, &name(slot, "ffn_down_shexp.weight"))?,
            }),
            moe: Some(Routed::Glu(import_moe(backend, file, slot)?)),
            block,
            ordinal,
            kv_rank,
            layer_scalar: 1.0,
            ple: None,
        });
        tracing::debug!(slot, kind = ?kind, "imported a layer slot from a gguf");
    }
    Ok(Weights {
        token_embd: embedding(backend, file)?,
        output_norm: vector(backend, file, "output_norm.weight")?,
        // Untied in this container: `output.weight` is its own tensor. A file
        // that tied them would have no `output.weight` at all, which is the
        // shape the dense family's reader handles and this one refuses - a
        // 248,320-row projection read from the embedding table is a different
        // arithmetic, not a fallback.
        lm_head: Some(weight(backend, file, "output.weight")?),
        ple_model_proj: None,
        ple_proj_norm: None,
        layers,
    })
}

/// One recurrent slot's weights.
///
/// Nine tensors and no permutation; the module header says why.
fn import_gdn(backend: &Arc<dyn Backend>, file: &ModelFile, slot: usize) -> CoreResult<Gdn> {
    Ok(Gdn {
        qkv: weight(backend, file, &name(slot, "attn_qkv.weight"))?,
        gate: weight(backend, file, &name(slot, "attn_gate.weight"))?,
        alpha: weight(backend, file, &name(slot, "ssm_alpha.weight"))?,
        beta: weight(backend, file, &name(slot, "ssm_beta.weight"))?,
        out: weight(backend, file, &name(slot, "ssm_out.weight"))?,
        conv1d: vector(backend, file, &name(slot, "ssm_conv1d.weight"))?,
        // `-exp(A_log)`, computed on the host: the kernels take the
        // recurrence coefficient itself, and the file stores its log.
        a: neg_exp(backend, file, &name(slot, "ssm_a"))?,
        dt_bias: vector(backend, file, &name(slot, "ssm_dt.bias"))?,
        norm: vector(backend, file, &name(slot, "ssm_norm.weight"))?,
    })
}

/// One attention slot's weights.
fn import_attn(backend: &Arc<dyn Backend>, file: &ModelFile, slot: usize) -> CoreResult<Attn> {
    Ok(Attn {
        q: weight(backend, file, &name(slot, "attn_q.weight"))?,
        // Every attention slot of this family owns its cache, so all three of
        // the key half's `Option`s are filled.
        k: Some(weight(backend, file, &name(slot, "attn_k.weight"))?),
        v: Some(weight(backend, file, &name(slot, "attn_v.weight"))?),
        output: weight(backend, file, &name(slot, "attn_output.weight"))?,
        q_norm: Some(vector(backend, file, &name(slot, "attn_q_norm.weight"))?),
        k_norm: Some(vector(backend, file, &name(slot, "attn_k_norm.weight"))?),
    })
}

/// One slot's routed feed-forward.
fn import_moe(backend: &Arc<dyn Backend>, file: &ModelFile, slot: usize) -> CoreResult<Moe> {
    Moe::new(
        weight(backend, file, &name(slot, "ffn_gate_inp.weight"))?,
        expert_slab(backend, file, &name(slot, "ffn_gate_exps.weight"))?,
        expert_slab(backend, file, &name(slot, "ffn_up_exps.weight"))?,
        expert_slab(backend, file, &name(slot, "ffn_down_exps.weight"))?,
        weight(backend, file, &name(slot, "ffn_gate_inp_shexp.weight"))?,
        ActDtype::F32,
    )
}

/// One projection of every expert in a slot, as the container already stacks
/// it.
///
/// One allocation and one copy. The tensor's memory order is
/// `[expert][row][element]`, which is what the expert entries stride through -
/// `expert * out_dim * row_blocks + row * row_blocks` - so no restacking is
/// needed and none is done.
fn expert_slab(backend: &Arc<dyn Backend>, file: &ModelFile, name: &str) -> CoreResult<ExpertSlab> {
    let desc = find(file, name)?;
    let codec = expert_codec(desc.dtype, name)?;
    let bytes = file.tensor_bytes(name).map_err(open_failure)?;
    let buffer = shiso_core::alloc(backend, u64::try_from(bytes.len()).unwrap_or(0))?;
    backend.copy_h2d(buffer.view(), &bytes)?;
    Ok(ExpertSlab::Block {
        weights: buffer,
        codec,
    })
}

/// The expert codec a stored format is, or the refusal naming the tensor.
///
/// The three `ggml` super-block formats this tree has routed entries for. A
/// fourth is a missing kernel rather than a missing branch, and it says so.
fn expert_codec(dtype: DType, name: &str) -> CoreResult<ExpertCodec> {
    match dtype {
        DType::Q4_K => Ok(ExpertCodec::Q4K),
        DType::Q5_K => Ok(ExpertCodec::Q5K),
        DType::Q6_K => Ok(ExpertCodec::Q6K),
        DType::Q8_0 => Ok(ExpertCodec::Q8_0),
        other => Err(CoreError::Unsupported {
            what: "a routed expert stack in a format this build has no expert entry for",
            backend: name_of(other, name),
        }),
    }
}

/// Names the format and the tensor in a refusal's backend slot.
///
/// `CoreError::Unsupported::backend` is a `&'static str`, so the pair cannot go
/// in it; the format alone can, and it is the half an operator acts on.
fn name_of(dtype: DType, name: &str) -> &'static str {
    tracing::error!(tensor = name, dtype = dtype.type_name(), "no expert entry");
    dtype.type_name()
}

/// The format the embedding table is stored in.
///
/// Read from the tensor rather than assumed, because the row gather is
/// dispatched on it: `UD-Q4_K_XL` keeps `token_embd` in `Q8_0` where the
/// `modelopt` path widens a `BF16` table to `f16`, and a gather dispatched for
/// the wrong one reads the right number of bytes from the right offset and
/// hands the stack a plausible embedding of the wrong token.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] when the table is absent.
pub fn embed_dtype(file: &ModelFile) -> CoreResult<DType> {
    Ok(find(file, EMBED)?.dtype)
}

/// The embedding table, which the row gather reads in place.
fn embedding(backend: &Arc<dyn Backend>, file: &ModelFile) -> CoreResult<Weight> {
    weight(backend, file, EMBED)
}

/// Imports one projection, checking its format against this family's set.
fn weight(backend: &Arc<dyn Backend>, file: &ModelFile, name: &str) -> CoreResult<Weight> {
    let desc = find(file, name)?;
    if !WEIGHT_TYPES.contains(&desc.dtype) {
        tracing::error!(tensor = name, dtype = desc.dtype.type_name(), "bad format");
        return Err(CoreError::Unsupported {
            what: "a weight stored in a format this family's gguf loader does not accept",
            backend: "qwen35moe",
        });
    }
    let dims = desc.shape.dims();
    let in_dim = extent(dims.first().copied().unwrap_or(0))?;
    let out_dim = extent(dims.get(1).copied().unwrap_or(1))?;
    let (host, range) = file.tensor_host(desc).map_err(open_failure)?;
    Weight::import(backend, desc.dtype, in_dim, out_dim, host, range)
}

/// Imports one norm or bias vector, which this container stores as `f32`.
fn vector(backend: &Arc<dyn Backend>, file: &ModelFile, name: &str) -> CoreResult<DeviceBuffer> {
    let desc = find(file, name)?;
    if desc.dtype != DType::F32 {
        tracing::error!(tensor = name, dtype = desc.dtype.type_name(), "not f32");
        return Err(CoreError::Unsupported {
            what: "a norm or bias vector stored as anything but single precision",
            backend: "qwen35moe",
        });
    }
    let (host, range) = file.tensor_host(desc).map_err(open_failure)?;
    import_host(backend, host, range)
}

/// `-exp(x)` over an `f32` vector, which is what the recurrence coefficient is.
///
/// The file stores `A_log`; every kernel here takes `A`. Computed on the host
/// because it is thirty-two values per slot and a launch to do it would be a
/// kernel that exists for the load path alone.
fn neg_exp(backend: &Arc<dyn Backend>, file: &ModelFile, name: &str) -> CoreResult<DeviceBuffer> {
    let desc = find(file, name)?;
    if desc.dtype != DType::F32 {
        tracing::error!(tensor = name, dtype = desc.dtype.type_name(), "not f32");
        return Err(CoreError::Unsupported {
            what: "a recurrence coefficient stored as anything but single precision",
            backend: "qwen35moe",
        });
    }
    let raw = file.tensor_bytes(name).map_err(open_failure)?;
    let values: Vec<u8> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|word| (-f32::from_le_bytes(*word).exp()).to_le_bytes())
        .collect();
    let buffer = shiso_core::alloc(backend, u64::try_from(values.len()).unwrap_or(0))?;
    backend.copy_h2d(buffer.view(), &values)?;
    Ok(buffer)
}

/// The directory entry for `name`, or the refusal naming it.
fn find<'f>(file: &'f ModelFile, name: &str) -> CoreResult<&'f TensorDesc> {
    inner(file)?
        .find(name)
        .ok_or_else(|| CoreError::ConfigMissing {
            key: name.to_string(),
        })
}

/// Builds a slot-scoped tensor name.
fn name(slot: usize, suffix: &str) -> String {
    format!("blk.{slot}.{suffix}")
}

/// Narrows one file dimension to the width a launch counts in.
fn extent(dim: u64) -> CoreResult<u32> {
    u32::try_from(dim).map_err(|_| CoreError::ShapeMismatch {
        field: "tensor extent",
        expected: u64::from(u32::MAX),
        got: dim,
    })
}

/// Presents a container failure in the core vocabulary.
fn open_failure(source: GgufError) -> CoreError {
    CoreError::Backend {
        backend: "qwen35moe",
        op: "load",
        source: Box::new(source),
    }
}
