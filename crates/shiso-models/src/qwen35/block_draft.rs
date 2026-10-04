//! The block drafter this family's dense checkpoints ship beside them.
//!
//! A directory with a `config.json` whose architecture is the block drafter's
//! and one `bfloat16` safetensors file: five layers at the target's residual
//! width, an input projection over the target's residual at the layers the
//! config names, and the selector's two codebooks. It is read here and handed
//! to the engine as [`BlockDraftParts`]; nothing about it is this family's
//! except the names.
//!
//! Every projection is quantized at load to 4-bit affine in groups of 64
//! ([`shiso_engine::quantize`]), so it runs on the same matmul as the target's
//! packed weights at an eighth of the bytes a `bfloat16` matrix would stream.
//! A drafter only proposes, so the precision this costs is paid in acceptance,
//! never in the answer.
//!
//! The same drafter also ships as one GGUF file (architecture `dflash`), its
//! projections already block-quantized. That container is read as stored -
//! the projections stay in their `q4_K`/`q6_K` blocks and its selector
//! codebooks stay `q4_K`, decoded a row at a time as candidates are scored -
//! with one translation: the file counts its taps from the embedding, so
//! hidden state `n` there is the output of layer `n - 1` here.

use std::path::Path;
use std::sync::Arc;

use half::bf16;
use serde::Deserialize;
use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, alloc};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::TokenId;
use shiso_engine::attn::Attn;
use shiso_engine::block_draft::{
    BlockDraftLayer, BlockDraftParts, BlockDraftShape, Codebook, CodebookRows, DynConv,
};
use shiso_engine::ffn::Ffn;
use shiso_engine::projection::Weight;
use shiso_engine::quantize::quantize_affine;
use shiso_modelfile::st_header::{RawTensor, Shard, StDType};

/// The architecture a drafter directory's config names.
pub const ARCHITECTURE: &str = "DFlash2DraftModel";

/// Bits the drafter's projections are quantized to at load.
const DRAFT_BITS: u32 = 4;

/// The drafter's own section of its config.
#[derive(Debug, Deserialize)]
struct DraftSection {
    block_size: u32,
    conv_group_size: u32,
    conv_kernel_size: u32,
    mask_token_id: u32,
    selector_rank: u32,
    selector_top_k: u32,
    target_layer_ids: Vec<u32>,
}

/// Rotary settings.
#[derive(Debug, Deserialize)]
struct Rope {
    rope_theta: f32,
}

/// The config document of a drafter directory.
#[derive(Debug, Deserialize)]
struct DraftConfig {
    architectures: Vec<String>,
    dflash_config: DraftSection,
    head_dim: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_attention_heads: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    num_target_layers: u32,
    rms_norm_eps: f32,
    rope_parameters: Rope,
    sliding_window: u32,
    vocab_size: u32,
}

/// The `general.architecture` of the drafter's GGUF container.
pub const GGUF_ARCHITECTURE: &str = "dflash";

/// Whether `path` holds a block drafter this loader reads: a directory with
/// the drafter's config, or a GGUF file of its architecture.
#[must_use]
pub fn is_block_drafter(path: &Path) -> bool {
    if path.is_file() {
        return shiso_modelfile::GgufFile::open(path)
            .is_ok_and(|file| file.metadata().architecture() == Some(GGUF_ARCHITECTURE));
    }
    let dir = path;
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .is_some_and(|doc| {
            doc["architectures"]
                .as_array()
                .is_some_and(|a| a.iter().any(|v| v.as_str() == Some(ARCHITECTURE)))
        })
}

/// The tensors of the drafter's one safetensors file, by name.
struct Tensors {
    shard: Shard,
    table: std::collections::HashMap<String, RawTensor>,
}

impl Tensors {
    fn open(dir: &Path) -> CoreResult<Self> {
        let (shard, list) =
            Shard::open_untagged(&dir.join("model.safetensors")).map_err(open_failure)?;
        Ok(Self {
            shard,
            table: list.into_iter().collect(),
        })
    }

    /// A `bfloat16` tensor widened to `f32`, with its shape.
    fn f32(&self, name: &str) -> CoreResult<(Vec<f32>, Vec<u64>)> {
        let (bits, shape) = self.bf16(name)?;
        Ok((
            bits.iter()
                .map(|&b| f32::from(bf16::from_bits(b)))
                .collect(),
            shape,
        ))
    }

    /// A `bfloat16` tensor as its bit patterns, with its shape.
    fn bf16(&self, name: &str) -> CoreResult<(Vec<u16>, Vec<u64>)> {
        let raw = self
            .table
            .get(name)
            .ok_or_else(|| CoreError::InvalidValue {
                key: "block drafter tensor",
                value: format!("{name} is not in the drafter's safetensors file"),
            })?;
        if raw.parsed_dtype(name).map_err(open_failure)? != StDType::Bf16 {
            return Err(CoreError::InvalidValue {
                key: "block drafter tensor",
                value: format!("{name} is {}, where the drafter ships bfloat16", raw.dtype),
            });
        }
        let bytes = self
            .shard
            .payload(raw.data_offsets[0], raw.data_offsets[1])
            .map_err(open_failure)?;
        Ok((
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect(),
            raw.shape.clone(),
        ))
    }
}

/// A shard error, as the engine's error type.
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

/// Reads the drafter in `dir` and puts it on the device.
///
/// `target_hidden` and `target_layers` are the target's residual width and
/// depth, which the drafter's input projection and its taps are checked
/// against; `target_vocab` is the vocabulary its logits come out over.
///
/// # Errors
///
/// [`CoreError::InvalidValue`] for a directory that is not a drafter this
/// loader reads or a tensor it does not carry, [`CoreError::ShapeMismatch`] for
/// one built for another target, and [`CoreError::Backend`] when an upload
/// fails.
pub fn load(
    backend: &Arc<dyn Backend>,
    dir: &Path,
    target: (u32, u32, u32),
) -> CoreResult<BlockDraftParts> {
    if dir.is_file() {
        return crate::qwen35::block_draft_gguf::load(backend, dir, target);
    }
    let (target_hidden, target_layers, target_vocab) = target;
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|err| {
        CoreError::InvalidValue {
            key: "block drafter config",
            value: format!("{}: {err}", dir.display()),
        }
    })?;
    let cfg: DraftConfig = serde_json::from_str(&text)?;
    if !cfg.architectures.iter().any(|a| a == ARCHITECTURE) {
        return Err(CoreError::InvalidValue {
            key: "block drafter architecture",
            value: format!(
                "{:?}, where this loader reads {ARCHITECTURE}",
                cfg.architectures
            ),
        });
    }
    for (field, want, got) in [
        (
            "block drafter num_target_layers",
            target_layers,
            cfg.num_target_layers,
        ),
        ("block drafter vocab_size", target_vocab, cfg.vocab_size),
        ("block drafter hidden_size", target_hidden, cfg.hidden_size),
    ] {
        if want != got {
            return Err(CoreError::ShapeMismatch {
                field,
                expected: u64::from(want),
                got: u64::from(got),
            });
        }
    }
    let d = &cfg.dflash_config;
    let shape = BlockDraftShape {
        hidden: cfg.hidden_size,
        n_head: cfg.num_attention_heads,
        n_head_kv: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        intermediate: cfg.intermediate_size,
        block: d.block_size,
        window: cfg.sliding_window,
        conv_taps: d.conv_kernel_size,
        conv_group: d.conv_group_size,
        rope_theta: cfg.rope_parameters.rope_theta,
        eps: cfg.rms_norm_eps,
        taps: d.target_layer_ids.clone(),
        mask_token: TokenId::new(d.mask_token_id),
        selector_rank: d.selector_rank,
        selector_top_k: d.selector_top_k,
    };
    shape.check(target_hidden, target_layers)?;
    let t = Tensors::open(dir)?;
    let clock = std::time::Instant::now();
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
    for i in 0..cfg.num_hidden_layers {
        layers.push(load_layer(backend, &t, i)?);
    }
    let codebook = |name: &str| -> CoreResult<Codebook> {
        let (values, dims) = t.bf16(name)?;
        let rank = u32::try_from(*dims.get(1).unwrap_or(&0)).unwrap_or(0);
        if rank != shape.selector_rank {
            return Err(CoreError::ShapeMismatch {
                field: "block drafter codebook rank",
                expected: u64::from(shape.selector_rank),
                got: u64::from(rank),
            });
        }
        Ok(Codebook {
            rank,
            rows: CodebookRows::Bf16(values),
        })
    };
    let parts = BlockDraftParts {
        fc: linear(backend, &t, "fc.weight")?,
        hidden_norm: vector(backend, &t, "hidden_norm.weight")?,
        layers,
        norm: vector(backend, &t, "norm.weight")?,
        hidden_projection: linear(backend, &t, "candidate_selector.hidden_projection.weight")?,
        predecessor: codebook("candidate_selector.predecessor_codebook")?,
        successor: codebook("candidate_selector.successor_codebook")?,
        shape,
    };
    tracing::info!(
        seconds = clock.elapsed().as_secs_f64(),
        dir = %dir.display(),
        "block drafter quantized and uploaded"
    );
    Ok(parts)
}

/// Drafter layer `i`.
fn load_layer(backend: &Arc<dyn Backend>, t: &Tensors, i: u32) -> CoreResult<BlockDraftLayer> {
    let name = |s: &str| format!("layers.{i}.{s}");
    Ok(BlockDraftLayer {
        input_norm: vector(backend, t, &name("input_layernorm.weight"))?,
        post_norm: vector(backend, t, &name("post_attention_layernorm.weight"))?,
        attn: Attn {
            q: linear(backend, t, &name("self_attn.q_proj.weight"))?,
            k: Some(linear(backend, t, &name("self_attn.k_proj.weight"))?),
            v: Some(linear(backend, t, &name("self_attn.v_proj.weight"))?),
            output: linear(backend, t, &name("self_attn.o_proj.weight"))?,
            q_norm: Some(vector(backend, t, &name("self_attn.q_norm.weight"))?),
            k_norm: Some(vector(backend, t, &name("self_attn.k_norm.weight"))?),
        },
        ffn: Ffn {
            gate: Some(linear(backend, t, &name("mlp.gate_proj.weight"))?),
            up: linear(backend, t, &name("mlp.up_proj.weight"))?,
            down: linear(backend, t, &name("mlp.down_proj.weight"))?,
        },
        attn_conv: DynConv {
            base: vector(backend, t, &name("attention_conv.base_kernel"))?,
            projection: linear(backend, t, &name("attention_conv.kernel_projection.weight"))?,
        },
        mlp_conv: DynConv {
            base: vector(backend, t, &name("mlp_conv.base_kernel"))?,
            projection: linear(backend, t, &name("mlp_conv.kernel_projection.weight"))?,
        },
    })
}

/// A `[out, in]` projection, quantized to [`DRAFT_BITS`].
fn linear(backend: &Arc<dyn Backend>, t: &Tensors, name: &str) -> CoreResult<Weight> {
    let (values, shape) = t.f32(name)?;
    let [rows, cols] = shape.as_slice() else {
        return Err(CoreError::InvalidValue {
            key: "block drafter projection",
            value: format!("{name} has shape {shape:?}, where a projection is two-dimensional"),
        });
    };
    let (rows, cols) = (
        usize::try_from(*rows).unwrap_or(0),
        usize::try_from(*cols).unwrap_or(0),
    );
    let packed = quantize_affine(&values, rows, cols, DRAFT_BITS)?;
    Weight::import_affine_slices(
        backend,
        DType::MlxAffine4Gs64,
        u32::try_from(cols).unwrap_or(u32::MAX),
        u32::try_from(rows).unwrap_or(u32::MAX),
        &packed.words,
        &packed.scales,
        &packed.biases,
    )
}

/// A norm weight or a convolution kernel, widened to `f32` on the device.
fn vector(backend: &Arc<dyn Backend>, t: &Tensors, name: &str) -> CoreResult<DeviceBuffer> {
    let (values, _) = t.f32(name)?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = alloc(backend, u64::try_from(bytes.len()).unwrap_or(0))?;
    backend.copy_h2d(buf.view(), &bytes)?;
    Ok(buf)
}
