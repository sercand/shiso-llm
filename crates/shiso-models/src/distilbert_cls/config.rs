//! What `config.json` says the classifier is, checked against what this
//! family can run.
//!
//! The file also carries a `calibration` object (temperature, thresholds).
//! Calibration is the caller's policy, applied to the logits this family
//! returns, so it is deliberately not read here: a threshold baked into the
//! model reader would be a second place for the decision to drift.

use std::collections::BTreeMap;

use serde::Deserialize;

use super::ClassifyError;

/// The `model_type` a document must declare.
pub const MODEL_TYPE: &str = "distilbert";

/// The number of classes this family's head has.
pub const NUM_LABELS: usize = 3;

/// The classifier's geometry, as `config.json` declares it.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifierConfig {
    /// Transformer layers.
    pub layers: usize,
    /// Attention heads.
    pub heads: usize,
    /// Residual width.
    pub dim: usize,
    /// Feed-forward width.
    pub hidden_dim: usize,
    /// Rows of the word table.
    pub vocab: usize,
    /// Rows of the position table, the longest sequence the model accepts.
    pub max_positions: usize,
    /// Class names, ordered by class id.
    pub labels: Vec<String>,
    /// Every `LayerNorm`'s epsilon.
    pub eps: f32,
}

/// The fields of `config.json` this family reads; everything else in the
/// document is ignored.
#[derive(Debug, Deserialize)]
struct Raw {
    model_type: String,
    n_layers: usize,
    n_heads: usize,
    dim: usize,
    hidden_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    num_labels: usize,
    id2label: BTreeMap<String, String>,
    layer_norm_eps: f64,
}

fn bad(what: impl Into<String>) -> ClassifyError {
    ClassifyError::BadConfig(what.into())
}

impl ClassifierConfig {
    /// Reads and checks a `config.json` document.
    ///
    /// # Errors
    ///
    /// [`ClassifyError::BadConfig`] naming the field that is absent,
    /// malformed, or describes something this family cannot run: another
    /// `model_type`, a zero dimension, a width the heads do not divide, or a
    /// label set other than three classes numbered 0 to 2.
    pub fn from_json(text: &str) -> Result<Self, ClassifyError> {
        let raw: Raw = serde_json::from_str(text).map_err(|e| bad(format!("config.json: {e}")))?;
        if raw.model_type != MODEL_TYPE {
            return Err(bad(format!(
                "model_type is {:?}, not {MODEL_TYPE:?}",
                raw.model_type
            )));
        }
        for (field, value) in [
            ("n_layers", raw.n_layers),
            ("n_heads", raw.n_heads),
            ("dim", raw.dim),
            ("hidden_dim", raw.hidden_dim),
            ("vocab_size", raw.vocab_size),
            ("max_position_embeddings", raw.max_position_embeddings),
        ] {
            if value == 0 {
                return Err(bad(format!("{field} is zero")));
            }
        }
        if !raw.dim.is_multiple_of(raw.n_heads) {
            return Err(bad(format!(
                "dim not divisible by heads: {} / {}",
                raw.dim, raw.n_heads
            )));
        }
        if raw.num_labels != NUM_LABELS {
            return Err(bad(format!(
                "num_labels is {}; this family's head has {NUM_LABELS}",
                raw.num_labels
            )));
        }
        if raw.id2label.len() != raw.num_labels {
            return Err(bad(format!(
                "id2label has {} entries, num_labels is {}",
                raw.id2label.len(),
                raw.num_labels
            )));
        }
        let labels = (0..raw.num_labels)
            .map(|id| {
                raw.id2label
                    .get(&id.to_string())
                    .cloned()
                    .ok_or_else(|| bad(format!("id2label has no entry for class {id}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (i, label) in labels.iter().enumerate() {
            if labels[..i].contains(label) {
                return Err(bad(format!("id2label repeats the label {label:?}")));
            }
        }
        if !(raw.layer_norm_eps.is_finite() && raw.layer_norm_eps > 0.0) {
            return Err(bad(format!(
                "layer_norm_eps is {}, not a positive number",
                raw.layer_norm_eps
            )));
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "an epsilon near 1e-12 is representable in f32"
        )]
        let eps = raw.layer_norm_eps as f32;
        Ok(Self {
            layers: raw.n_layers,
            heads: raw.n_heads,
            dim: raw.dim,
            hidden_dim: raw.hidden_dim,
            vocab: raw.vocab_size,
            max_positions: raw.max_position_embeddings,
            labels,
            eps,
        })
    }

    /// Width of one attention head.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }
}
