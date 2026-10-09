//! Qwen-Image-2.1 text-to-image.
//!
//! The family conditions a diffusion transformer on the hidden states of a
//! Qwen3-VL decoder read over a templated prompt. So far it holds the prompt
//! ([`prompt`]): the template, the injection-safe encoding and the rows the
//! transformer never sees. The text encoder itself is the shared
//! `crate::diffusion::qwen3_text`, run to its last layer with the final norm
//! left off. [`vae`] is the RGBA decoder that turns the transformer's
//! 64-channel latent tokens into the final image.

pub(crate) mod prompt;
pub(crate) mod vae;

#[cfg(test)]
mod encoder_tests;
#[cfg(test)]
mod prompt_tests;
#[cfg(test)]
mod vae_tests;
