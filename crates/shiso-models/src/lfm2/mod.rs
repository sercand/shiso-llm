//! The LFM2 hybrid family: gated short-convolution slots interleaved with
//! grouped-query attention slots, each followed by a SwiGLU feed-forward, from
//! a GGUF container.
//!
//! [`config`] reads the geometry out of the container and checks the document
//! against it, and [`model`] makes the checkpoint resident over the engine's
//! dense kernels and its short-convolution block. A decision checkpoint of
//! this family is the same stack read differently; the config says which.

mod ckpt;
pub mod config;
mod model;

pub use ckpt::LFM2_MODEL_ID;
pub use config::{Lfm2Config, Lfm2Readout};
pub use model::Lfm2Model;

use crate::catalog::FamilyFacts;

/// Family facts for LFM2.
pub const FACTS: FamilyFacts = FamilyFacts { name: "lfm2" };
