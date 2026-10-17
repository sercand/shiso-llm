//! The FLUX.3 text encoder: the shared Qwen3-style prefill tapped after
//! eight layers, over a prompt padded and masked the way the reference pads.
//!
//! The reference renders the chat template around the prompt, pads the ids
//! on the right to a multiple of 80 (at most 4096) with the pad token, runs
//! the decoder with the pad rows masked out of attention and rotated at
//! position 1, and keeps the pad rows' hidden states: the transformer is
//! trained to see them as text tokens. FLUX.2's encoder truncates instead;
//! this one never does - a prompt over the limit is refused.
//!
//! The taps are the reference's `hidden_states[4, 8, ..., 32]`, counting
//! `hidden_states[0]` as the embedding output, i.e. the stream after
//! 0-indexed layers 3, 7, ..., 31. Layers past the last tap are neither
//! loaded nor run.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the pipeline that drives the encoder lands in a later task; the expectation \
                  fails the build once it does, which is the cue to delete it"
    )
)]

use std::path::{Path, PathBuf};

use shiso_vision::generator::ImageGenError;

use super::pack::padded_len;
use crate::diffusion::ops::Ops;
use crate::diffusion::prompt::{PromptTokenizer, render_user_turn};
use crate::diffusion::qwen3_text::{TapSet, TextEncoder, TextEncoderDesc};
use crate::diffusion::weights::StStore;

/// The 0-indexed layers whose outputs condition the transformer.
pub(crate) const TAP_LAYERS: [usize; 8] = [3, 7, 11, 15, 19, 23, 27, 31];

/// The decoder's tensor-name prefixes an export may use: the vision-language
/// checkpoint nests the language model, a text-only export does not.
pub(crate) const PREFIXES: [&str; 3] = ["model.language_model.", "model.", "language_model.model."];

/// A prompt encoded to conditioning rows.
#[derive(Debug)]
pub(crate) struct EncodedText {
    /// Row-major `(len, 8 * hidden)` tapped hidden states, pad rows included.
    pub tokens: Vec<f32>,
    /// Rows: the padded length.
    pub len: usize,
    /// Rows that are real prompt tokens (the rest are pad rows).
    pub real: usize,
}

/// A prompt as the encoder reads it: ids padded to the multiple of 80.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PaddedPrompt {
    /// The padded ids: the rendered prompt, then pad ids.
    pub ids: Vec<u32>,
    /// How many leading ids are the prompt's own.
    pub real: usize,
}

/// The tensor-name prefix the decoder in `store` lives under.
pub(crate) fn detect_prefix(
    store: &StStore,
    files: &[PathBuf],
) -> Result<&'static str, ImageGenError> {
    let found: Vec<&'static str> = PREFIXES
        .iter()
        .copied()
        .filter(|prefix| store.contains(&format!("{prefix}embed_tokens.weight")))
        .collect();
    match found.as_slice() {
        [one] => Ok(one),
        [] => Err(ImageGenError::Checkpoint(format!(
            "no text encoder embedding found: looked for '<prefix>embed_tokens.weight' with \
             prefix {} in {}",
            PREFIXES.map(|p| format!("'{p}'")).join(", "),
            files
                .iter()
                .map(|f| f.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
        several => Err(ImageGenError::Checkpoint(format!(
            "the text encoder files hold an embedding under more than one prefix ({}); \
             which decoder is meant is not guessed",
            several.join(", ")
        ))),
    }
}

/// Opens the decoder in `files` (described by `config`) and uploads every
/// weight through the last tap in its stored format.
///
/// The tensor prefix is found from the names, so a vision-language export
/// and a text-only one both load; two prefixes matching is refused.
///
/// # Errors
/// [`ImageGenError::Checkpoint`] when no known prefix holds the decoder or
/// the weights do not reach the last tap.
pub(crate) fn load(
    files: &[PathBuf],
    config: &Path,
    ops: &Ops,
) -> Result<TextEncoder, ImageGenError> {
    let prefix = detect_prefix(&StStore::open_files(files)?, files)?;
    TextEncoder::load(
        &TextEncoderDesc {
            weights: files,
            config,
            prefix,
            taps: TapSet::Layers(TAP_LAYERS.to_vec()),
        },
        ops,
    )
}

/// Renders the chat template around `prompt`, tokenizes it and pads it.
///
/// # Errors
/// [`ImageGenError::EmptyPrompt`] for a prompt of only whitespace,
/// [`ImageGenError::PromptTooLong`] over 4096 tokens (refused, never cut),
/// and a checkpoint refusal when the tokenizer names no pad token.
pub(crate) fn tokenize(
    tok: &PromptTokenizer,
    template: &str,
    prompt: &str,
) -> Result<PaddedPrompt, ImageGenError> {
    if prompt.trim().is_empty() {
        return Err(ImageGenError::EmptyPrompt);
    }
    let (head, tail) = render_user_turn(template)?;
    let mut ids = tok.encode_template(&head)?;
    ids.extend(tok.encode_caller(prompt)?);
    ids.extend(tok.encode_template(&tail)?);
    let real = ids.len();
    let len = padded_len(real)?;
    ids.resize(len, tok.pad_id()?);
    Ok(PaddedPrompt { ids, real })
}

/// Encodes `prompt` to the `(len, 8 * hidden)` conditioning the transformer
/// reads, `len` being the padded length.
///
/// # Errors
/// The refusals of [`tokenize`] and of the encoder's prefill.
pub(crate) fn encode(
    enc: &TextEncoder,
    tok: &PromptTokenizer,
    template: &str,
    ops: &Ops,
    prompt: &str,
) -> Result<EncodedText, ImageGenError> {
    let padded = tokenize(tok, template, prompt)?;
    let tokens = enc.prefill_padded(ops, &padded.ids, padded.real)?;
    Ok(EncodedText {
        tokens,
        len: padded.ids.len(),
        real: padded.real,
    })
}
