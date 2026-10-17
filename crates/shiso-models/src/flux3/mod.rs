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
//! [`config`] locates a checkpoint and inspects its headers; the helper
//! modules hold the pure parts (schedule, ids, token packing) that can be
//! checked without a device.

pub mod config;

use crate::catalog::FamilyFacts;

/// Family facts for FLUX.3 image generation.
pub const FACTS: FamilyFacts = FamilyFacts { name: "flux3" };

#[cfg(test)]
mod config_tests;
