//! The routed family's multi-token-prediction head, from a directory of its
//! own.
//!
//! ```text
//! Qwen3.6-35B-A3B-MTP-4bit/
//!   config.json          model_type qwen3_5_mtp, quantization {bits 4, group_size 64, mode affine}
//!   model.safetensors    46 tensors
//! ```
//!
//! **A head is a separate file here, not a block of the target's container.**
//! That is how this family's head is published - `mlx-community` splits it out
//! of `Qwen/Qwen3.6-35B-A3B` for `mlx-vlm`'s speculative decoding - and it is
//! also the shape the dense family already had, through `mtp_file` in its
//! sidecar's document. The target's own GGUF carries a `NextN` block at
//! `blk.40` as well; a caller that has one of the two does not need the other.
//!
//! # What makes this head cheap to serve
//!
//! Its codec is **MLX affine 4-bit, group 64** - which `ExpertCodec::Q4Affine`
//! already has graded routed entries for, and `DType::MlxAffine4Gs64` already
//! has a matvec for. So the head needed no new kernel: what it needed was for
//! `DraftParts` to be able to carry a routed feed-forward, which is a field
//! rather than a codec.
//!
//! Its experts also arrive **already stacked**: `mlp.switch_mlp.gate_proj` is
//! one `[experts, out_dim, in_dim / 8]` tensor, which is the
//! `[expert][row][element]` order `ExpertSlab` wants. The target's `modelopt`
//! export stores 256 separate tensors per projection and the loader stacks
//! them; nothing here has to.
//!
//! # The head is the BASE model's, and that is the pairing rule
//!
//! These weights were split from `Qwen/Qwen3.6-35B-A3B`, so they predict the
//! hidden states of that model. `unsloth`'s `UD-Q4_K_XL` is a quantization of
//! the same base and pairs with them. A FINETUNE of that base does not: its
//! hidden states are its own, so this head over a fine-tune's body would draft
//! fluently and be rejected. Nothing here can check
//! that - the files do not say what they were trained against - so it is stated
//! rather than enforced, and the acceptance rate is what reveals a mispairing.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use half::bf16;
use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, alloc};
use shiso_core::error::{CoreError, CoreResult};
use shiso_engine::attn::Attn;
use shiso_engine::draft::{DraftParts, HeadNorm};
use shiso_engine::ffn::Ffn;
use shiso_engine::layers::Routed;
use shiso_engine::moe::{ActDtype, ExpertSlab, Moe};
use shiso_engine::projection::Weight;
use shiso_engine::shape::{HybridShape, MoeShape};
use shiso_modelfile::st_header::{RawTensor, Shard};

/// The container a head directory holds its weights in.
const CONTAINER: &str = "model.safetensors";

/// Tensors a head of this family ships.
///
/// Checked rather than trusted, for the reason the dense head's reader gives:
/// a container holding a different number is a different head, and finding
/// every tensor this module looks for does not rule that out.
const HEAD_TENSORS: usize = 46;

/// The codec every packed tensor of this head is stored in.
///
/// Read from the head's own `config.json` by the caller and asserted here
/// rather than taken as an argument: this reader indexes words, scales and
/// biases from the geometry, and a head packed at another group size would be
/// read at the right length from the wrong stride.
const PACKED: DType = DType::MlxAffine4Gs64;

/// One head, mapped, with its tensors indexed by name.
pub struct MtpHead {
    /// The container, mapped for as long as this lives.
    shard: Shard,
    /// Every tensor the container holds.
    table: HashMap<String, RawTensor>,
}

impl std::fmt::Debug for MtpHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The mapping itself has no host-readable content worth printing, and
        // its length is the one fact about it a reader acts on.
        f.debug_struct("MtpHead")
            .field("mapped", &self.shard.len())
            .field("tensors", &self.table.len())
            .finish()
    }
}

impl MtpHead {
    /// Opens the head directory at `dir`.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the container cannot be mapped or its header
    /// does not parse, and [`CoreError::ShapeMismatch`] when it holds a tensor
    /// count other than the one this head is.
    pub fn open(dir: &Path) -> CoreResult<Self> {
        let (shard, raw) = Shard::open_untagged(&dir.join(CONTAINER)).map_err(read_failure)?;
        let table: HashMap<String, RawTensor> = raw.into_iter().collect();
        if table.len() != HEAD_TENSORS {
            return Err(CoreError::ShapeMismatch {
                field: "draft head tensor count",
                expected: HEAD_TENSORS as u64,
                got: table.len() as u64,
            });
        }
        Ok(Self { shard, table })
    }

    /// Builds the parts a [`shiso_engine::draft::DraftHead`] is made resident
    /// from.
    ///
    /// `shape` is the TARGET's, and every width below is checked against it:
    /// the head is a second model over the same residual stream, and one built
    /// at another width would bind into launches sized by the target's geometry
    /// and succeed at every one.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] naming the first tensor the container does
    /// not hold, [`CoreError::ShapeMismatch`] when a tensor's shape disagrees
    /// with the target's widths, and whatever the device reports.
    pub fn parts(&self, backend: &Arc<dyn Backend>, shape: &HybridShape) -> CoreResult<DraftParts> {
        let moe = shape.moe.ok_or(CoreError::Unsupported {
            what: "a routed draft head over a target that has no routed block",
            backend: "qwen35moe",
        })?;
        let hidden = shape.hidden;
        let layer = "layers.0";
        Ok(DraftParts {
            // `[hidden, 2 * hidden]`: the one projection of the head that reads
            // a concatenation, and the reason `DraftHead::new` checks widths.
            fc: self.packed(backend, "fc", 2 * hidden, hidden)?,
            norm_embed: self.norm(backend, "pre_fc_norm_embedding.weight")?,
            norm_hidden: self.norm(backend, "pre_fc_norm_hidden.weight")?,
            input_norm: self.norm(backend, &format!("{layer}.input_layernorm.weight"))?,
            ffn_norm: self.norm(backend, &format!("{layer}.post_attention_layernorm.weight"))?,
            output_norm: self.norm(backend, "norm.weight")?,
            head_norm: HeadNorm::Rms,
            hidden_prenorm: None,
            attn: self.attn(backend, layer, shape)?,
            // The SHARED expert, which is the routed block's dense half and
            // takes the same field the target's `LayerWeights::ffn` does.
            ffn: Ffn {
                gate: Some(self.packed(
                    backend,
                    &format!("{layer}.mlp.shared_expert.gate_proj"),
                    hidden,
                    shape.intermediate,
                )?),
                up: self.packed(
                    backend,
                    &format!("{layer}.mlp.shared_expert.up_proj"),
                    hidden,
                    shape.intermediate,
                )?,
                down: self.packed(
                    backend,
                    &format!("{layer}.mlp.shared_expert.down_proj"),
                    shape.intermediate,
                    hidden,
                )?,
            },
            moe: Some((Routed::Glu(self.routed(backend, layer, hidden, moe)?), moe)),
        })
    }

    /// The head's single attention block.
    fn attn(
        &self,
        backend: &Arc<dyn Backend>,
        layer: &str,
        shape: &HybridShape,
    ) -> CoreResult<Attn> {
        let widths = *shape.attn.first().ok_or(CoreError::ShapeMismatch {
            field: "attention slots the head's block is shaped like",
            expected: 1,
            got: 0,
        })?;
        let hidden = shape.hidden;
        // The query projection carries the output gate, so it is twice the head
        // run wide - the same rule the target's own attention slots follow.
        let q_out = widths.q_heads * widths.head_dim * if shape.attn_output_gate { 2 } else { 1 };
        let kv_out = widths.kv_heads * widths.head_dim;
        let at = |role: &str| format!("{layer}.self_attn.{role}");
        Ok(Attn {
            q: self.packed(backend, &at("q_proj"), hidden, q_out)?,
            k: Some(self.packed(backend, &at("k_proj"), hidden, kv_out)?),
            v: Some(self.packed(backend, &at("v_proj"), hidden, kv_out)?),
            output: self.packed(
                backend,
                &at("o_proj"),
                widths.q_heads * widths.head_dim,
                hidden,
            )?,
            q_norm: Some(self.norm(backend, &at("q_norm.weight"))?),
            k_norm: Some(self.norm(backend, &at("k_norm.weight"))?),
        })
    }

    /// The head's routed feed-forward.
    fn routed(
        &self,
        backend: &Arc<dyn Backend>,
        layer: &str,
        hidden: u32,
        moe: MoeShape,
    ) -> CoreResult<Moe> {
        let at = |role: &str| format!("{layer}.mlp.{role}");
        Moe::new(
            self.packed(backend, &at("gate"), hidden, moe.experts)?,
            self.slab(backend, &at("switch_mlp.gate_proj"))?,
            self.slab(backend, &at("switch_mlp.up_proj"))?,
            self.slab(backend, &at("switch_mlp.down_proj"))?,
            self.packed(backend, &at("shared_expert_gate"), hidden, 1)?,
            ActDtype::F32,
        )
    }

    /// One projection of every expert, as the container already stacks it.
    ///
    /// Three buffers and no restacking: the words, the per-group scales and the
    /// per-group biases each arrive as one `[experts, out_dim, ...]` tensor,
    /// which is the stride the routed entries walk.
    fn slab(&self, backend: &Arc<dyn Backend>, name: &str) -> CoreResult<ExpertSlab> {
        Ok(ExpertSlab::Q4Affine {
            weights: self.upload(backend, &format!("{name}.weight"))?,
            scales: self.upload(backend, &format!("{name}.scales"))?,
            biases: self.upload(backend, &format!("{name}.biases"))?,
        })
    }

    /// One packed projection, checked against the widths it is launched at.
    fn packed(
        &self,
        backend: &Arc<dyn Backend>,
        name: &str,
        in_dim: u32,
        out_dim: u32,
    ) -> CoreResult<Weight> {
        // `[out_dim, in_dim / 8]`: eight four-bit weights to a 32-bit word. The
        // row count is checked here and the word count by
        // `import_affine_slices`, which indexes from the geometry rather than
        // from the array it was handed.
        let words = self.tensor(&format!("{name}.weight"))?;
        let rows = words.shape.first().copied().unwrap_or(0);
        if rows != u64::from(out_dim) {
            return Err(CoreError::ShapeMismatch {
                field: "draft head projection rows",
                expected: u64::from(out_dim),
                got: rows,
            });
        }
        Weight::import_affine_slices(
            backend,
            PACKED,
            in_dim,
            out_dim,
            self.bytes(&format!("{name}.weight"))?,
            self.bytes(&format!("{name}.scales"))?,
            self.bytes(&format!("{name}.biases"))?,
        )
    }

    /// One norm vector, widened from `BF16` to the `f32` every norm entry reads.
    fn norm(&self, backend: &Arc<dyn Backend>, name: &str) -> CoreResult<DeviceBuffer> {
        let raw = self.bytes(name)?;
        let values: Vec<u8> = raw
            .chunks_exact(2)
            .flat_map(|pair| {
                bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]]))
                    .to_f32()
                    .to_le_bytes()
            })
            .collect();
        let buffer = alloc(backend, u64::try_from(values.len()).unwrap_or(0))?;
        backend.copy_h2d(buffer.view(), &values)?;
        Ok(buffer)
    }

    /// One tensor's bytes, uploaded verbatim.
    fn upload(&self, backend: &Arc<dyn Backend>, name: &str) -> CoreResult<DeviceBuffer> {
        let raw = self.bytes(name)?;
        let buffer = alloc(backend, u64::try_from(raw.len()).unwrap_or(0))?;
        backend.copy_h2d(buffer.view(), raw)?;
        Ok(buffer)
    }

    /// The directory entry for `name`, or the refusal naming it.
    fn tensor(&self, name: &str) -> CoreResult<&RawTensor> {
        self.table
            .get(name)
            .ok_or_else(|| CoreError::ConfigMissing {
                key: name.to_string(),
            })
    }

    /// One tensor's bytes.
    fn bytes(&self, name: &str) -> CoreResult<&[u8]> {
        let t = self.tensor(name)?;
        self.shard
            .payload(t.data_offsets[0], t.data_offsets[1])
            .map_err(read_failure)
    }
}

/// Presents a container failure in the core vocabulary.
fn read_failure(source: shiso_modelfile::GgufError) -> CoreError {
    CoreError::Backend {
        backend: "qwen35moe-mtp",
        op: "load",
        source: Box::new(source),
    }
}
