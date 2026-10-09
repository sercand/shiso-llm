//! The dense-hybrid Qwen3.5 family: what its config means, what its tensors are
//! called, and the model type that ties the two to the engine.
//!
//! This crate is thin on purpose. It holds the three things that are true of
//! this checkpoint family and of nothing else:
//!
//! - [`Qwen35Config`] reads the family's config subtree and checks every
//!   dimension against what the kernels were written for;
//! - [`bind_qwen35`] maps the family's tensor names onto layer roles;
//! - [`Qwen35Model`] imports those tensors and answers [`shiso_core::Model`], and
//!   tells the engine's checkpoint tier which tag and which quantization its stored
//!   bytes were written under ([`QWEN35_MODEL_ID`], [`quant_bits_of`]);
//! - [`Qwen35Kernels`] issues the two blocks' launches, which are this family's
//!   kernels and nobody else's;
//! - [`Qwen35Tower`] is the family's vision tower - its dimensions, manifest,
//!   encoder and loader all live under [`vision`] with it, because a tower is
//!   a fact about this family and of nothing else.
//!
//! Everything else - the residual stream, the block families, the paged cache,
//! the sessions, the sampler - is in the engine, because none of it is a fact
//! about this family. A second family is a second crate like this one, and it
//! should be able to land without editing a line here or anywhere below.
//!
//! # The `qwen3_5_moe` variant does NOT belong here, and what changed the ruling
//!
//! This section previously read *"belongs here, and what it would cost"*, and
//! ruled that a `Qwen3_5MoeForConditionalGeneration` checkpoint is this family
//! with a routed feed-forward. **That ruling is overturned and
//! [`crate::qwen35moe`] is where it now lives.** The evidence it was taken on
//! was not wrong; it was evidence about the *kernels*, and it was read as
//! evidence about the family.
//!
//! What still holds, and is why the two modules are not independent: the
//! recurrent geometry IS this family's exactly - 32 value heads, 16 key heads,
//! a 128-wide head, a 4-tap convolution - and the attention block IS this
//! family's gated one at 16 heads over `head_dim` 256 with an output gate
//! folded into a doubled `q_proj`. [`crate::qwen35moe`] dispatches
//! [`Qwen35Kernels`] for both rather than owning a copy, so those two blocks
//! cannot drift between the families.
//!
//! What did not hold is everything above the kernels. 94,393 tensors over
//! three safetensors shards at `F8_E4M3` and `U8`, one tensor per expert, two
//! config documents where this family has one - none of that is this family's
//! container or this family's loader, and llama.cpp carries `LLM_ARCH_QWEN35`
//! and `LLM_ARCH_QWEN35MOE` as separate architecture ids with separate model
//! files for the same reason.
//!
//! [`Qwen35Config`] asserts 32 layers, hidden 2560, 4 key/value heads and a
//! dense `intermediate_size`, so it refuses that document today - and it now
//! refuses it on the architecture name, before a width is consulted, because
//! the two families spell `model_type` differently.
//!
//! **The shape is shared, and that is a separate ruling from this one.**
//! [`shiso_engine::shape::HybridShape`] gained `moe: Option<MoeShape>` rather
//! than the new module getting a shape of its own; the argument is on that
//! type. Nothing dense reads the field, and this family states `None`.
//!
//! [`MoeShape`]: shiso_engine::shape::MoeShape

mod binding;
pub mod block_draft;
mod block_draft_gguf;
mod ckpt;
mod config;
mod kernels;
mod model;
pub mod modelopt;
pub mod mtp;
pub mod rotation;
pub mod st_names;
pub mod vision;

pub use binding::{
    AttnRoles, AttnWeights, BoundLayer, DescBinding, FfnRoles, FfnWeights, GdnRoles, GdnWeights,
    LayerRoles, ModelBinding, ModelRoles, VECTOR_TYPES, WEIGHT_TYPES, bind_qwen35,
    bind_qwen35_desc,
};
pub use ckpt::{QWEN35_MODEL_ID, quant_bits_of};
pub use config::Qwen35Config;
pub use kernels::Qwen35Kernels;
pub use model::Qwen35Model;
pub use mtp::{
    AffineQuant, HeadTensor, MtpAttn, MtpLayer, MtpManifest, MtpMlp, MtpWeights, PackedMatrix,
};
pub use st_names::Qwen35Names;
pub use vision::Qwen35Tower;

use crate::catalog::FamilyFacts;

/// Family facts for Qwen3.5.
pub const FACTS: FamilyFacts = FamilyFacts { name: "qwen35" };
