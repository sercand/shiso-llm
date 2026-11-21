//! Inkling-2 looped multimodal Mixture-of-Experts family.
//!
//! Supports the official `inkling2_loop` checkpoint format (24 physical layers
//! with a 3-pass looped middle core giving 48 logical layers, mixed NVFP4 and
//! MXFP8 routed experts, relative-bias attention with per-query sinks, block
//! attention residuals, static/dynamic short convolutions, native `ViT` vision
//! encoder, and `DSpark` / MTP block speculator).

pub mod config;
pub mod model;
pub mod mtp;
pub mod ops;
pub mod vision;
pub mod weights;

pub use config::{Inkling2Config, LoopConfig, MtpConfig, TextConfig, VisionConfig};
pub use model::Inkling2Model;
pub use mtp::Inkling2MtpDrafter;
pub use vision::Inkling2VisionTower;

use crate::catalog::FamilyFacts;

/// Catalog entry for the Inkling-2 family.
pub const FACTS: FamilyFacts = FamilyFacts { name: "inkling2" };
