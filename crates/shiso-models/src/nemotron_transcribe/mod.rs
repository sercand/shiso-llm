//! The speech encoder of the Nemotron speech-to-text checkpoint: a speech
//! transformer and a speaker transformer over the same mel frames, fused into
//! the rows its decoder reads where the prompt's audio placeholder sits, and
//! the timestamp head that times the decoder's words.
//!
//! The decoder is [`crate::nemotron_h`]'s GGUF flavour; this module is the
//! encoder export beside it ([`config`], [`weights`]), the frontend and window
//! plan ([`frontend`]), the shared rotary transformer ([`transformer`]), the
//! speaker branch with its cross-window cache ([`diar`]) and the fusion
//! ([`encoder`]), and on the text side the card's instructions ([`presets`])
//! and the cleanup of what the decoder wrote ([`postprocess`]).

pub mod config;
pub mod diar;
pub mod encoder;
pub mod frontend;
pub mod pairing;
pub mod postprocess;
pub mod presets;
pub mod tower;
pub mod transformer;
pub mod weights;

pub use config::{BlockDims, PeeConfig};
pub use encoder::{Encoded, encode};
pub use pairing::{PLACEMENT, SPEECH_MODEL_KEY, speech_encoder_for};
pub use postprocess::{collapse_loops, speaker_turns, strip_tags};
pub use presets::{Preset, PresetError, parse_preset};
pub use tower::NemotronTranscribeTower;
pub use weights::PeeWeights;

use crate::catalog::FamilyFacts;

/// Family facts for the speech encoder.
pub const FACTS: FamilyFacts = FamilyFacts {
    name: "nemotron_transcribe",
};

#[cfg(test)]
#[path = "encoder_tests.rs"]
mod encoder_tests;
