//! The LFM2 hybrid family: gated short-convolution slots interleaved with
//! grouped-query attention slots, each followed by a gated feed-forward, from
//! a GGUF container - and the d1 decision checkpoint built on it.
//!
//! [`config`] reads the geometry out of the container and checks the document
//! against it, and [`model`] makes the checkpoint resident over the engine's
//! dense kernels and its short-convolution block. A decision checkpoint of
//! this family is the same stack read differently; the config says which, and
//! [`decision`] holds its answer-slot readout - its prompt, its option codes
//! and the spellings an answer is read from - which the server reads its
//! decision routes through. [`vision`] is the family's vision tower, the
//! SigLIP2-style encoder and the projector of its multimodal container.

mod ckpt;
pub mod config;
pub mod decision;
mod model;
pub mod vision;

pub use ckpt::LFM2_MODEL_ID;
pub use config::{Lfm2Config, Lfm2Readout};
pub use decision::D1Readout;
pub use model::Lfm2Model;
pub use vision::Lfm2Tower;

use crate::catalog::FamilyFacts;

/// Family facts for LFM2.
pub const FACTS: FamilyFacts = FamilyFacts { name: "lfm2" };
