//! The host side of audio: bytes to PCM, PCM to log-mel frames, and model
//! output back to text.
//!
//! Like the vision crate, this one stops at the device boundary. The encoders
//! that consume mel frames live with each family in `shiso-models`, because
//! they are written against one family's graph. What is shared is everything
//! that is the same whatever model listens: decoding a request's bytes to
//! 16 kHz mono, the log-mel frontends, vocabularies of subword pieces and
//! their detokenization, and transcript assembly.
//!
//! # Decoding
//!
//! [`decode_audio`] reads WAV itself, always. Every other container goes down
//! a chain of decoders, each present only if the build (and, for ffmpeg, the
//! operator) asked for it:
//!
//! 1. **symphonia**, with the `decode-native` feature: mp3, AAC in mp4/m4a,
//!    FLAC, Vorbis and PCM, in process.
//! 2. **ffmpeg**, with the `decode-ffmpeg` feature and a binary named in the
//!    [`AudioDecodePolicy`]: everything else it decodes, Opus among them.
//! 3. Otherwise [`AudioError::UnsupportedContainer`].
//!
//! A step passes the bytes on only when it does not read that container or
//! codec; bytes that break a format a step reads are refused as
//! [`AudioError::Malformed`]. The crate has no default features, so a library
//! dependent never switches a decoder on behind the server's back.
#![forbid(unsafe_code)]

pub mod chunks;
mod context;
mod decode;
mod diarizer;
mod error;
mod fft;
mod mel;
mod mel_stream;
#[cfg(feature = "decode-native")]
mod native;
mod num;
mod pcm;
mod recognizer;
mod resample;
mod tower;
pub mod transcript;
mod vocab;
mod wav;

pub use context::{Context, ContextEncoder, ContextIds, is_echo, keyword_words};
pub use decode::{AudioDecodePolicy, DecodeLimits, decode_audio, sniff_container};
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
