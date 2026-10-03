//! The sparse mixture-of-experts `Qwen4Exp` family: what its metadata means, what
//! its tensors are called, and the model type that ties the two to the engine.
//!
//! This crate is thin on purpose. It holds the three things that are true of
//! this checkpoint family and of nothing else:
//!
//! - [`Qwen4ExpConfig`] carries the fixed shape table the kernels were written
//!   for, and checks a container's own metadata against every number in it;
//! - [`bind_qwen4exp`] maps the family's tensor names onto layer roles;
//! - [`Qwen4ExpModel`] imports those tensors and answers [`shiso_core::Model`],
//!   and tells the engine's checkpoint tier which tag and which quantization its
//!   stored bytes were written under ([`QWEN4EXP_MODEL_ID`], [`quant_bits_of`]).
//!
//! Everything else - the residual streams, the routed feed-forward, the
//! indexer, the paged cache, the sessions, the sampler - is in the engine,
//! because none of it is a fact about this family. A third family with the same
//! blocks is a third crate like this one, and it should be able to land without
//! editing a line there or anywhere below.

mod binding;
mod ckpt;
mod config;
pub mod exl3;
pub mod fp8;
pub mod mlx;
mod model;

pub use binding::{
    AttnRoles, BlockRoles, BoundLayer, ExpertRoles, MixRoles, ModelBinding, NGRAM_BIAS,
    NGRAM_TABLE, NgramRoles, RecurrentRoles, bind_qwen4exp,
};
pub use ckpt::{QWEN4EXP_MODEL_ID, quant_bits_of};
pub use config::{
    ARCH, EXPERT_INTERMEDIATE, EXPERTS, EXPERTS_SHARED, EXPERTS_USED, FULL_ATTENTION_INTERVAL,
    GDN_CONV, GDN_HEAD_DIM, GDN_QK_HEADS, GDN_V_HEADS, HEAD_DIM, HIDDEN, INDEXER_COMPRESS,
    INDEXER_HEAD_DIM, INDEXER_HEADS, INDEXER_TOP_K, KV_HEADS, MROPE_HEIGHT, MROPE_TIME,
    MROPE_WIDTH, N_LAYERS, N_ROT, NgramHash, PLE_CONV, PLE_EOS_TOKEN, PLE_HEAD_DIM,
    PLE_HEADS_PER_NGRAM, PLE_LAYER, PLE_NGRAM, Q_HEADS, Qwen4ExpConfig, RMS_EPS,
    ROPE_ORIGINAL_CONTEXT, ROPE_THETA, STREAM_RANK, STREAMS, VOCAB, is_attention, layer_kinds,
    shape,
};
pub use model::Qwen4ExpModel;

use crate::catalog::FamilyFacts;

/// Family facts for `Qwen4Exp`.
pub const FACTS: FamilyFacts = FamilyFacts { name: "qwen4exp" };
