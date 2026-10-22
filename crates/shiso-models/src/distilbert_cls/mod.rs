//! The sequence-classification family: a six-layer post-norm encoder whose
//! first position feeds a two-layer head, answering one of three labels.
//!
//! This module owns what is this family's alone: how a directory is
//! recognised ([`describes`]), the typed errors, and the config and weight
//! binding. Calibration of the logits is the caller's policy and is not read
//! from the checkpoint.

mod config;
mod weights;

#[cfg(test)]
mod tests;

use std::path::Path;

pub use config::{ClassifierConfig, MODEL_TYPE, NUM_LABELS};
pub use weights::{ClassifierWeights, HostLinear, HostTable, LayerWeights, NormWeights, validate};

use crate::catalog::FamilyFacts;

/// Family facts for the sequence classifier.
pub const FACTS: FamilyFacts = FamilyFacts {
    name: "distilbert_cls",
};

/// Why a classifier could not be opened or run.
#[derive(Debug, thiserror::Error)]
pub enum ClassifyError {
    /// No device could be opened to run the encoder on.
    #[error("no device: {0}")]
    NoDevice(String),
    /// `config.json` is malformed or describes a model this family cannot run.
    #[error("config.json: {0}")]
    BadConfig(String),
    /// The checkpoint lacks a tensor the config implies.
    #[error("missing tensor {0}")]
    MissingTensor(String),
    /// A tensor is present but misshapen, in an unread encoding, or truncated.
    #[error("tensor {name} {why}")]
    BadTensor {
        /// The tensor's name.
        name: String,
        /// What is wrong with it.
        why: String,
    },
    /// The tokenizer could not be loaded or run.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// The device failed while importing weights or running a pass.
    #[error("device: {0}")]
    Device(String),
    /// A local read failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Whether `dir` holds a checkpoint this family reads: a `config.json`
/// declaring [`MODEL_TYPE`] with a `model.safetensors` beside it.
#[must_use]
pub fn describes(dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    let is_ours = serde_json::from_str::<serde_json::Value>(&text)
        .is_ok_and(|v| v.get("model_type").and_then(|t| t.as_str()) == Some(MODEL_TYPE));
    is_ours && dir.join("model.safetensors").is_file()
}
