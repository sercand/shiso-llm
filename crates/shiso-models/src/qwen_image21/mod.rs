//! Qwen-Image-2.1 text-to-image.
//!
//! The family conditions a diffusion transformer on the hidden states of a
//! Qwen3-VL decoder read over a templated prompt. So far it holds the prompt
//! ([`prompt`]): the template, the injection-safe encoding and the rows the
//! transformer never sees. The text encoder itself is the shared
//! `crate::diffusion::qwen3_text`, run to its last layer with the final norm
//! left off.

pub(crate) mod prompt;

#[cfg(test)]
mod encoder_tests;
#[cfg(test)]
mod prompt_tests;
