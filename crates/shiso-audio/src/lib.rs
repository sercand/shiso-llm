//! The host side of audio: bytes to PCM, PCM to log-mel frames, and model
//! output back to text.
//!
//! Like the vision crate, this one stops at the device boundary. The encoders
//! that consume mel frames live with each family in `shiso-models`, because
//! they are written against one family's graph. What is shared is everything
//! that is the same whatever model listens: decoding a request's bytes to
//! 16 kHz mono, the log-mel frontends, vocabularies of subword pieces and
//! their detokenization, and transcript assembly.
#![forbid(unsafe_code)]

pub mod chunks;
mod decode;
mod diarizer;
mod error;
mod fft;
mod mel;
mod mel_stream;
mod num;
mod pcm;
mod recognizer;
mod resample;
mod tower;
pub mod transcript;
mod vocab;
mod wav;

pub use decode::{DecodeLimits, decode_audio, sniff_container};
pub use diarizer::{DiarizationPass, Diarizer};
pub use error::AudioError;
pub use mel::{
    Filterbank, Framing, LogScale, MelFrames, MelFrontend, MelScale, MelSpec, Preemphasis,
    Spectrum, Window, log10_max_floor,
};
pub use mel_stream::MelStream;
pub use pcm::{MODEL_RATE, Pcm};
pub use recognizer::{
    Language, OfflineBatch, OfflineJob, RecognizerStream, SpanResult, SpeechRecognizer,
    StreamOptions, TokenEvent,
};
pub use tower::{AudioClip, AudioTower, clip_for};
pub use vocab::{Detokenizer, PieceVocab};
