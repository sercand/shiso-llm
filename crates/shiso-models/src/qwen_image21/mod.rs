//! Qwen-Image-2.1 text-to-image.
//!
//! The family conditions a diffusion transformer on the hidden states of a
//! Qwen3-VL decoder read over a templated prompt. [`prompt`] holds the
//! template, the injection-safe encoding and the rows the transformer never
//! sees; the text encoder itself is the shared `crate::diffusion::qwen3_text`,
//! run to its last layer with the final norm left off. [`config`] reads the
//! install's documents and resolves the transformer's tensor names;
//! [`embed`] and [`rope`] are its timestep/modulation/input embedders and
//! three-axis rotary positions; [`transformer`] is the diffusion
//! transformer, split into a once-per-job text prefix and a per-step target
//! pass; [`scheduler`] builds the sigma ladder the steps walk. [`vae`] is the
//! RGBA decoder that turns the transformer's 64-channel latent tokens into
//! the final image. [`pipeline`] composes them into the served generator
//! and decides which stay on the device between jobs.

pub(crate) mod config;
pub(crate) mod embed;
pub(crate) mod pipeline;
pub(crate) mod prompt;
pub(crate) mod rope;
pub(crate) mod scheduler;
pub(crate) mod transformer;
pub(crate) mod vae;

use crate::catalog::FamilyFacts;

/// Family facts for Qwen-Image-2.1 image generation.
pub(crate) const FACTS: FamilyFacts = FamilyFacts {
    name: "qwen_image21",
};

#[cfg(test)]
mod dit_tests;
#[cfg(test)]
mod encoder_tests;
#[cfg(test)]
mod pipeline_tests;
#[cfg(test)]
mod prompt_tests;
#[cfg(test)]
mod transformer_tests;
#[cfg(test)]
mod vae_tests;
