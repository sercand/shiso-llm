//! The FLUX.2 text encoder: the shared Qwen3-style prefill
//! ([`crate::diffusion::qwen3_text`]) tapped after layers 8, 17 and 26.
//!
//! Those are the reference's `hidden_states[9]`, `[18]` and `[27]` counting
//! `hidden_states[0]` as the embedding output, i.e. the stream after
//! 0-indexed layers 8, 17, 26. The three hidden states are concatenated into
//! the (`seq_len`, 3 * hidden) conditioning the diffusion transformer reads.
//! Layers past the last tap are neither loaded nor evaluated: their output
//! reaches no consumer.
//!
//! The component home is a directory holding the decoder's safetensors and
//! its flat Qwen3 `config.json`; tensors are named `model.layers.<n>.*`.

use std::path::Path;

use shiso_vision::generator::ImageGenError;

use crate::diffusion::ops::Ops;
use crate::diffusion::qwen3_text::{TapSet, TextEncoder, TextEncoderDesc};
use crate::diffusion::weights::safetensors_files;

/// The residual-stream layers whose hidden states condition the transformer:
/// the reference taps `hidden_states[9]`, `[18]` and `[27]` with
/// `hidden_states[0]` being the embedding output, so the 0-indexed layers
/// after which those states exist are 8, 17 and 26.
const TAP_LAYERS: [usize; 3] = [8, 17, 26];

/// Opens the component at `path` (a directory with its `config.json`) and
/// uploads every weight through the last tap in its stored format.
pub(crate) fn load(path: &Path, ops: &Ops) -> Result<TextEncoder, ImageGenError> {
    let weights = safetensors_files(path)?;
    TextEncoder::load(
        &TextEncoderDesc {
            weights: &weights,
            config: &path.join("config.json"),
            prefix: "model.",
            taps: TapSet::Layers(TAP_LAYERS.to_vec()),
        },
        ops,
    )
}
