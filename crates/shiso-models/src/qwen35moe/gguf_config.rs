//! This family's geometry, read out of a GGUF file's own metadata.
//!
//! The sibling of [`super::config`], which reads the same widths out of a pair
//! of JSON documents beside a `modelopt` export. Both end at
//! [`Qwen35MoeConfig::new`], whose doc comment already anticipated this caller:
//! "a container that carries its dimensions as metadata rather than as a
//! document".
//!
//! # Why this exists
//!
//! The `modelopt` NVFP4 export this family was ported against is a **bad
//! quantization**: its body collapses onto one CJK token, which
//! `scripts/qwen35moe_mtp_reference.py` establishes in `transformers` with no
//! part of this tree involved, and for which llama.cpp's fluent output from a
//! GGUF of the same base model is the control. Reading the GGUF is what gives
//! this architecture a container that answers.
//!
//! # The two places a GGUF disagrees with the JSON documents
//!
//! **`block_count` counts the draft layer.** `unsloth`'s
//! `Qwen3.6-35B-A3B-UD-Q4_K_XL` reports 41 blocks for a 40-layer model, because
//! `blk.40` is the `NextN` block - a multi-token-prediction head that is not part
//! of the target's forward pass. `nextn_predict_layers` says how many of the
//! trailing blocks are that, and the target's layer count is the difference. A
//! reader that took `block_count` at face value would build a 41-slot stack,
//! find `blk.40`'s tensors, run them, and answer fluent nonsense - the draft
//! head is a *different function* over the same residual width, so every shape
//! checks out.
//!
//! **The vocabulary is not a metadata key.** It is the row count of
//! `token_embd.weight`, which is how every other GGUF reader in this tree finds
//! it too.
//!
//! # What is asserted rather than read
//!
//! The layer kinds. A GGUF states `full_attention_interval` and nothing else
//! about which slots are recurrent, where the JSON document carries an explicit
//! `layer_types` array. The two agree on this checkpoint - slot `i` is
//! attention exactly when `i % interval == interval - 1` - and that rule is
//! applied here rather than guessed at per slot, so a container whose blocks do
//! not match it is refused by the binder when a tensor it expects is absent
//! rather than mis-run.

use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::TokenId;
use shiso_engine::shape::{
    Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, MoeShape, NormScheme,
};
use shiso_modelfile::metadata::GgufMetadata;

use crate::qwen35moe::config::Qwen35MoeConfig;

/// The architecture name this reader accepts, as the file spells it.
pub const GGUF_ARCH: &str = "qwen35moe";

/// The end-of-sequence id this family uses when the file names none.
///
/// The same value the JSON path reads from `eos_token_id`. A GGUF carries it
/// under `tokenizer.ggml.eos_token_id`, which is read when present.
const DEFAULT_EOS: u32 = 248_044;

/// Reads the geometry out of a GGUF file's metadata and its embedding shape.
///
/// `vocab` is the row count of `token_embd.weight`; the caller has the tensor
/// directory and this module does not.
///
/// # Errors
///
/// [`CoreError::UnknownArch`] when the file declares another architecture, and
/// [`CoreError::ConfigMissing`] naming the key when one this family cannot
/// default is absent.
pub fn config_from_gguf(meta: &GgufMetadata, vocab: u32) -> CoreResult<Qwen35MoeConfig> {
    match meta.architecture() {
        Some(GGUF_ARCH) => {}
        other => {
            return Err(CoreError::UnknownArch {
                model_type: other.unwrap_or("<none>").to_string(),
            });
        }
    }
    let need_u32 = |suffix: &str| -> CoreResult<u32> {
        meta.arch_u32(suffix)
            .ok_or_else(|| CoreError::ConfigMissing {
                key: format!("{GGUF_ARCH}.{suffix}"),
            })
    };
    let scalar = |suffix: &str| -> CoreResult<f32> {
        meta.arch_f32(suffix)
            .ok_or_else(|| CoreError::ConfigMissing {
                key: format!("{GGUF_ARCH}.{suffix}"),
            })
    };

    let blocks = need_u32("block_count")?;
    // The trailing blocks that are the draft head rather than the model. Absent
    // means none, which is what a container without a head reports.
    let nextn = meta.arch_u32("nextn_predict_layers").unwrap_or(0);
    let n_layers = blocks
        .checked_sub(nextn)
        .ok_or_else(|| CoreError::InvalidValue {
            key: "qwen35moe.nextn_predict_layers",
            value: format!("{nextn} draft blocks in a file with {blocks} blocks"),
        })?;
    let interval = need_u32("full_attention_interval")?;
    if interval == 0 {
        return Err(CoreError::InvalidValue {
            key: "qwen35moe.full_attention_interval",
            value: "zero, which would make no slot an attention slot".to_string(),
        });
    }

    let hidden = need_u32("embedding_length")?;
    let heads = need_u32("attention.head_count")?;
    let kv_heads = need_u32("attention.head_count_kv")?;
    let head_dim = need_u32("attention.key_length")?;
    let n_rot = need_u32("rope.dimension_count")?;
    let rope_theta = scalar("rope.freq_base")?;
    let rms_eps = scalar("attention.layer_norm_rms_epsilon")?;

    // The recurrent block's widths. `ssm.inner_size` is the VALUE run's total
    // width and `ssm.state_size` one head's, so their ratio is the value head
    // count; `ssm.group_count` is the key head count, which is smaller.
    let inner = need_u32("ssm.inner_size")?;
    let state = need_u32("ssm.state_size")?;
    if state == 0 || inner % state != 0 {
        return Err(CoreError::InvalidValue {
            key: "qwen35moe.ssm.inner_size",
            value: format!("{inner} is not a whole number of {state}-wide heads"),
        });
    }
    let value_heads = inner / state;
    let key_heads = need_u32("ssm.group_count")?;
    let conv = need_u32("ssm.conv_kernel")?;

    let experts = need_u32("expert_count")?;
    let used = need_u32("expert_used_count")?;
    let routed_width = need_u32("expert_feed_forward_length")?;
    // The SHARED expert's width is the slot's dense feed-forward width, which is
    // what `HybridShape::intermediate` means. A file that states no shared width
    // has no shared expert, and this family always does.
    let shared_width = need_u32("expert_shared_feed_forward_length")?;

    let layers: Vec<LayerKind> = (0..n_layers)
        .map(|slot| {
            if slot % interval == interval - 1 {
                LayerKind::Attn
            } else {
                LayerKind::Gdn
            }
        })
        .collect();
    let attn_slots = layers.iter().filter(|k| **k == LayerKind::Attn).count();

    let shape = HybridShape {
        attn: vec![AttnLayer::new(heads, kv_heads, head_dim, n_rot, n_rot, rope_theta); attn_slots],
        layers,
        hidden,
        vocab,
        intermediate: shared_width,
        gdn: GdnShape::new(value_heads, key_heads, state, conv),
        // Every switch below is this family's, and the same value the JSON
        // reader sets. They are architecture rather than container, so a GGUF
        // that omitted one would not be describing this family.
        attn_output_gate: true,
        rms_eps,
        activation: Activation::SwiGlu,
        norms: NormScheme::PreOnly,
        embed_scale: None,
        final_softcap: None,
        layer_scalar: false,
        ple: None,
        media_positions: MediaPositions::Mrope3D,
        v_norm: false,
        moe: Some(MoeShape::new(experts, used, 1, routed_width)),
    };
    let eos = meta
        .get_u32("tokenizer.ggml.eos_token_id")
        .unwrap_or(DEFAULT_EOS);
    Ok(Qwen35MoeConfig::new(
        shape,
        TokenId::new(eos),
        interval,
        // Stated per tensor by the container rather than by a document beside
        // it, exactly as for an MLX checkpoint.
        None,
    ))
}

/// How many trailing blocks of a GGUF are the draft head.
///
/// Read by the head's loader, which needs the block index the JSON path has no
/// equivalent of.
#[must_use]
pub fn nextn_blocks(meta: &GgufMetadata) -> u32 {
    meta.arch_u32("nextn_predict_layers").unwrap_or(0)
}
