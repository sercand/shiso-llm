//! The routed-hybrid Qwen3.5 mixture-of-experts family: what its two config documents mean,
//! and what its weights are stored as.
//!
//! A third family module rather than a flag on [`crate::qwen35`], for the
//! reason llama.cpp carries `LLM_ARCH_QWEN35` and `LLM_ARCH_QWEN35MOE` as
//! separate architecture ids with separate model files: the container, the
//! config, the loader and the feed-forward all differ, and only the recurrent
//! and attention *kernels* are shared. This module dispatches
//! [`crate::qwen35::Qwen35Kernels`] for those two blocks rather than owning a
//! copy of them, so the two families cannot drift on the part they share.
//!
//! # What was ruled the other way, and why
//!
//! `crates/shiso-models/src/qwen35/mod.rs` recorded, on 2026-09-20, that this
//! checkpoint "is this family" - the dense hybrid with a routed feed-forward.
//! That reading held for the kernels and not for anything above them. The
//! recurrent geometry IS the dense family's exactly, the attention block IS its
//! gated one, and both are dispatched here unchanged; but 94,393 tensors over
//! three safetensors shards at `F8_E4M3` and `U8`, one tensor per expert, is
//! not that family's container and not its loader.
//!
//! # The shape, however, is shared, and that is a separate ruling
//!
//! [`shiso_engine::shape::HybridShape`] gained one field -
//! `moe: Option<MoeShape>` - rather than this module getting a shape of its
//! own. The argument is written on that type: this checkpoint is that skeleton
//! exactly, its only addition is one step of the layer loop, and a shape of its
//! own would have meant a second copy of the engine's layer loop and stack
//! whose only difference from the first was that one step.
//!
//! # The Splash package is a second container for the same family
//!
//! [`splash`] reads the two index files of a pre-quantized `*-Splash` package:
//! a `manifest.json` of every artifact with its digest, and a `layout.json`
//! saying what each `.bin` holds and where. It is a container, not a family -
//! the config, the tensor roles and the kernels above are unchanged by it -
//! which is why it is a submodule here rather than a fourth crate module. It
//! names no device and reads no weight; the loader that does is built on top
//! of what these two readers return.
//!
//! [`shiso_engine::shape::MoeShape`]: shiso_engine::shape::MoeShape

pub mod binding;
mod ckpt;
mod config;
pub mod gguf;
mod gguf_config;
mod model;
pub mod mtp;
mod quant;
pub mod splash;

pub use binding::{TEXT_ROOTS, check_export, expected_tensors};
pub use ckpt::{QWEN35MOE_MODEL_ID, quant_bits_of};
pub use config::Qwen35MoeConfig;
pub use gguf::{embed_dtype, import_weights_gguf, open_gguf};
pub use gguf_config::{GGUF_ARCH, config_from_gguf, nextn_blocks};
pub use model::Qwen35MoeModel;
pub use mtp::MtpHead;
pub use quant::{QuantAlgo, QuantConfig};

use crate::catalog::{FamilyFacts, FamilyFeatures};

/// Family facts for Qwen3.5 `MoE`.
pub const FACTS: FamilyFacts = FamilyFacts {
    name: "qwen35moe",
    accepted_roles: &["base", "draft", "draft_vocab", "vision"],
    tasks: &["chat"],
    features: FamilyFeatures {
        tools: true,
        reasoning_efforts: &["none", "low", "medium", "high"],
        default_reasoning_effort: Some("medium"),
        structured_output: true,
    },
};
