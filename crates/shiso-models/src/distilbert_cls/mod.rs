//! The sequence-classification family: a six-layer post-norm encoder whose
//! first position feeds a two-layer head, answering one of three labels.
//!
//! This module owns what is this family's alone: how a directory is
//! recognised ([`describes`]), the typed errors, and the config and weight
//! binding.
//!
//! # Usage
//!
//! The caller opens the device, once, and hands it in. Creating the backend
//! compiles the device kernels, so do it off any latency-sensitive thread and
//! keep the [`Classifier`] for the life of the process; it is `Sync`, and calls
//! from several threads are serialized on one lock.
//!
//! ```ignore
//! let backend: Arc<dyn Backend> = MetalBackend::new()?;
//! let classifier = Classifier::open(&ClassifierDesc { dir: model_dir, backend })?;
//! let logits = classifier.logits("git status")?;
//! for (label, logit) in classifier.labels().iter().zip(logits) {
//!     // `labels()` is in logit order: safe, needs-approval, dangerous.
//! }
//! ```
//!
//! # Input
//!
//! `text` is tokenized with the checkpoint's own `tokenizer.json`, which adds
//! the start and end markers and truncates to the first 128 ids; the last of
//! those is then the end marker, whatever was cut. Anything after the first 126
//! content tokens has no effect on the answer. Empty and whitespace-only text
//! is the two markers alone and yields finite logits. [`Classifier::token_ids`]
//! shows exactly what a text is classified on.
//!
//! # Output
//!
//! [`Classifier::logits`] returns raw logits in [`Classifier::labels`] order
//! (`safe`, `needs-approval`, `dangerous` for the shipped checkpoint).
//! Calibration is the caller's job: the checkpoint's `config.json` carries a
//! temperature, an energy threshold and a probability threshold, but nothing
//! here reads or applies them.
//!
//! # Errors
//!
//! [`ClassifyError`] is typed and nothing panics on bad input. `open` reports a
//! malformed config, a missing or misshapen tensor, or a broken tokenizer by
//! name before the first call. `open` also refuses a position table shorter
//! than the 128-id window and a tokenizer with more ids than the word table has
//! rows, so no input can index past either. `logits` reports
//! [`ClassifyError::Tokenizer`] or [`ClassifyError::Device`], and
//! [`ClassifyError::BadTensor`] if the head disagrees with the config (cannot
//! happen once `open` has validated the weights); a failed call leaves the
//! classifier usable for the next one. A machine with no device is the caller's state to report, since
//! the caller creates the backend.
//!
//! # CUDA
//!
//! A CUDA consumer must build with `SHISO_CUDA_COMPILE=1`; without it every
//! kernel fails to resolve at runtime and the build only warns. The device
//! tests in `tests.rs` are backend-neutral, but they have only been run on
//! Metal: the CUDA backend is untested for this family.
//!
//! # Cost
//!
//! Measured on an Apple M4 (macOS 26.5.2, Metal) in a release build by the
//! ignored `measure_latency` test
//! (`cargo test --release -p shiso-models --lib measure_latency -- --ignored --nocapture`):
//!
//! | what | time |
//! | --- | --- |
//! | backend creation, `Classifier::open` and the first `logits`, first run after a build | 2.19 s |
//! | backend creation, `Classifier::open` and the first `logits`, later runs | 0.08 s |
//! | median of 100 calls, 17 ids | 2.4 to 3.1 ms |
//! | median of 100 calls, 128 ids | 6.7 ms |
//!
//! The first-run figure most likely includes compiling the device kernel
//! library, which the platform then caches (the gap to later runs fits that,
//! but it was not isolated); a cold cache on another machine, or after a system
//! update, can take much longer, so create the backend once and not on a UI
//! thread.

mod config;
mod forward;
mod head;
mod tokens;
mod weights;

#[cfg(test)]
mod tests;

use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use shiso_core::Backend;

pub use config::{ClassifierConfig, MODEL_TYPE, NUM_LABELS};
use weights::ClassifierWeights;

use crate::audio_ops::Ops;
use crate::audio_ops::store::TensorStore;
use crate::catalog::FamilyFacts;
use forward::Encoder;

/// Family facts for the sequence classifier.
pub const FACTS: FamilyFacts = FamilyFacts {
    name: "distilbert_cls",
};

/// Why a classifier could not be opened or run.
#[derive(Debug, thiserror::Error)]
pub enum ClassifyError {
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

/// The `architectures` entry that says the checkpoint has a classification
/// head; a bare encoder of the same `model_type` has none.
pub const ARCHITECTURE: &str = "DistilBertForSequenceClassification";

/// Whether `dir` holds a checkpoint this family reads: a `config.json`
/// declaring [`MODEL_TYPE`] and [`ARCHITECTURE`] with a `model.safetensors`
/// beside it.
#[must_use]
pub fn describes(dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    let is_ours = serde_json::from_str::<serde_json::Value>(&text).is_ok_and(|v| {
        v.get("model_type").and_then(|t| t.as_str()) == Some(MODEL_TYPE)
            && v.get("architectures")
                .and_then(|a| a.as_array())
                .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(ARCHITECTURE)))
    });
    is_ours && dir.join("model.safetensors").is_file()
}

/// A count as `f32`: exact for every width and head size here.
#[expect(
    clippy::cast_precision_loss,
    reason = "widths stay far below 2^24, where f32 is exact"
)]
pub(super) fn f32_of(n: usize) -> f32 {
    n as f32
}

/// What a call would otherwise trip over later: a sequence longer than the
/// position table, or an id past the word table.
fn check_limits(cfg: &ClassifierConfig, vocab: usize) -> Result<(), ClassifyError> {
    if cfg.max_positions < tokens::MAX_TOKENS {
        return Err(ClassifyError::BadConfig(format!(
            "max_position_embeddings is {}, below the {} ids a call can carry",
            cfg.max_positions,
            tokens::MAX_TOKENS
        )));
    }
    if vocab > cfg.vocab {
        return Err(ClassifyError::BadConfig(format!(
            "the tokenizer has {vocab} ids but vocab_size is {}",
            cfg.vocab
        )));
    }
    Ok(())
}

/// What [`Classifier::open`] needs.
#[derive(Debug, Clone)]
pub struct ClassifierDesc<'a> {
    /// The model directory: `config.json`, `model.safetensors`, `tokenizer.json`.
    pub dir: &'a Path,
    /// The device to run on. The caller opens it, so a missing device is the
    /// caller's state to report.
    pub backend: Arc<dyn Backend>,
}

/// A loaded classifier.
///
/// Calls are serialized on one lock: the device scratch is shared, and a
/// failed call leaves the lock usable for the next.
#[derive(Debug)]
pub struct Classifier {
    labels: Vec<String>,
    tokens: tokens::WordPiece,
    encoder: Mutex<Encoder>,
}

impl Classifier {
    /// Loads the checkpoint in `desc.dir` onto the device.
    ///
    /// # Errors
    ///
    /// [`ClassifyError::Io`] for an unreadable `config.json` or
    /// `model.safetensors`, [`ClassifyError::BadConfig`] for a position table
    /// shorter than 128 ids or a tokenizer wider than the word table, then
    /// whatever tokenizer or weight loading reports.
    pub fn open(desc: &ClassifierDesc<'_>) -> Result<Self, ClassifyError> {
        let text = std::fs::read_to_string(desc.dir.join("config.json"))?;
        // Before any upload: a broken tokenizer.json fails fast.
        let tokens = tokens::WordPiece::load(&desc.dir.join("tokenizer.json"))?;
        let cfg = ClassifierConfig::from_json(&text)?;
        check_limits(&cfg, tokens.vocab_size())?;
        let weights_path = desc.dir.join("model.safetensors");
        // An absent or unreadable file is an I/O failure, not a bad tensor.
        std::fs::File::open(&weights_path)?;
        let store =
            TensorStore::open_safetensors(&weights_path).map_err(|e| ClassifyError::BadTensor {
                name: "model.safetensors".to_owned(),
                why: e.to_string(),
            })?;
        let ops = Ops::new(Arc::clone(&desc.backend));
        let weights = ClassifierWeights::load(&ops, &store, &cfg)?;
        desc.backend.finish_imports();
        Ok(Self {
            labels: cfg.labels.clone(),
            tokens,
            encoder: Mutex::new(Encoder::new(ops, cfg, weights)),
        })
    }

    /// Class names in logit order.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The ids `text` is classified on: at most 128, markers included.
    ///
    /// # Errors
    ///
    /// [`ClassifyError::Tokenizer`] when encoding fails.
    pub fn token_ids(&self, text: &str) -> Result<Vec<u32>, ClassifyError> {
        self.tokens.ids(text)
    }

    /// Raw logits in [`Self::labels`] order for the first 128 tokens of
    /// `text`. Calibration is the caller's.
    ///
    /// # Errors
    ///
    /// [`ClassifyError::Tokenizer`] or [`ClassifyError::Device`];
    /// [`ClassifyError::BadTensor`] if the head disagrees with the config,
    /// which [`Self::open`] has already ruled out.
    pub fn logits(&self, text: &str) -> Result<[f32; NUM_LABELS], ClassifyError> {
        let ids = self.tokens.ids(text)?;
        let mut encoder = self.encoder.lock();
        let cls = encoder.cls_vector(&ids)?;
        let w = encoder.weights();
        let out = head::apply(
            &w.pre_classifier.weight,
            &w.pre_classifier.bias,
            &w.classifier.weight,
            &w.classifier.bias,
            &cls,
            (encoder.config().dim, NUM_LABELS),
        )?;
        <[f32; NUM_LABELS]>::try_from(out).map_err(|o| ClassifyError::BadTensor {
            name: "classifier.weight".to_owned(),
            why: format!("gave {} logits, not {NUM_LABELS}", o.len()),
        })
    }
}
