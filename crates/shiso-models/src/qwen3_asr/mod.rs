//! Speech recognizer of one audio-LLM architecture: a windowed transformer
//! audio tower over 1-second conv-stem chunks, whose rows replace the audio
//! placeholders of a chat prompt, and a dense decoder with a tied head that
//! writes `language X<asr_text>transcript`.
//!
//! Two checkpoints share the architecture - the base model and a fine-tune
//! with a retrained decoder - and every container they ship in (HF, MLX, a
//! GGUF text file with its mmproj) reads through one name table and one
//! config reader ([`config`]). The facts behind every constant are in the
//! model notes (`docs/superpowers/specs/2026-09-26-qwen3-asr-model-notes.md`),
//! cited by section where each is used.

pub mod config;
pub mod encoder;
mod names;
pub mod weights;
