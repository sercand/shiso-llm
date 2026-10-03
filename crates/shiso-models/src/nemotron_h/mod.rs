//! The hybrid state-space family: selective state-space blocks, attention
//! blocks with no positional encoding, and routed squared-rectifier
//! feed-forwards, from an MLX 4-bit export.
//!
//! [`config`] turns the document into the engine's stack shape - pairing each
//! routed layer with the mixing layer before it into one slot - and [`model`]
//! makes the checkpoint resident over the engine's parametric kernels.

mod ckpt;
pub mod config;
mod model;

pub use ckpt::NEMOTRON_H_MODEL_ID;
pub use config::{Mixer, NemotronHConfig, SlotLayers};
pub use model::NemotronHModel;

use crate::catalog::FamilyFacts;

/// Family facts for Nemotron-H.
pub const FACTS: FamilyFacts = FamilyFacts { name: "nemotron_h" };
