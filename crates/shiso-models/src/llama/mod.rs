//! The dense llama family: grouped-query attention slots that rotate their
//! heads without normalizing them, each followed by a gated feed-forward, with
//! the logits tied to the embedding - read from a Clef GGUF, with the joint
//! schema head beside it.
//!
//! [`config`] reads the geometry out of the container and checks the document
//! against it, and [`model`] makes the backbone resident over the engine's
//! dense kernels and loads [`crate::clef::ClefHead`] to read its final rows.
//! The container is the Clef one only: [`config`] says why a stock export of
//! this family is refused.

mod ckpt;
pub mod config;
#[cfg(test)]
mod config_tests;
mod model;

pub use ckpt::LLAMA_MODEL_ID;
pub use config::LlamaConfig;
pub use model::LlamaModel;

use crate::catalog::FamilyFacts;

/// Family facts for the dense llama family.
pub const FACTS: FamilyFacts = FamilyFacts { name: "llama" };
