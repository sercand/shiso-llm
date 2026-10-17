//! FLUX.3 Klein (distilled) image generation.
//!
//! This is not FLUX.2 with more blocks. The transformer keeps one token
//! stream per mode (`image`, `image_cond`, `text`), gives each mode its own
//! first blocks and its own modulation, and only then runs a shared
//! single-stream tail; the sampler is a flow-map Euler walk with a second
//! time embedding for the step's target time. The family reads its numbers
//! off the checkpoint's tensors and refuses a checkpoint whose shapes do not
//! match what the kernels were built for, naming the tensor.
//!
//! [`pipeline`] composes them; [`vae`] is the autoencoder, [`reference`] prepares edit
//! pictures and [`sampler`] is the Euler walk.
//!
//! [`config`] locates a checkpoint and inspects its headers; the helper
//! modules hold the pure parts (schedule, ids, token packing) that can be
//! checked without a device.

pub mod config;
pub(crate) mod pack;
pub(crate) mod pipeline;
pub(crate) mod reference;
pub use reference::default_size_for_references;
pub(crate) mod sampler;
pub(crate) mod scheduler;
pub(crate) mod text_encoder;
pub(crate) mod transformer;
pub(crate) mod vae;

use crate::catalog::FamilyFacts;

/// Family facts for FLUX.3 image generation.
pub const FACTS: FamilyFacts = FamilyFacts { name: "flux3" };

#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod pack_tests;
#[cfg(test)]
mod pipeline_tests;
#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod sampler_tests;
#[cfg(test)]
mod scheduler_tests;
#[cfg(test)]
mod text_encoder_tests;
#[cfg(test)]
mod transformer_tests;
#[cfg(test)]
mod vae_tests;
