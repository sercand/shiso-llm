//! `hf_quant_config.json`: what each of this checkpoint's tensors is stored as.
//!
//! A second document beside `config.json` because it answers a different
//! question. `config.json` says the stack is 40 layers of 2048; this says that
//! 130 of its projections are FP8 E4M3 with FP8 activations, that 161 of them -
//! every expert, every shared expert and the output projection - are NVFP4 at a
//! group size of 16, that the key/value cache was calibrated for FP8, and that
//! the draft head is excluded from all of it.
//!
//! # A target is a prefix, not a tensor
//!
//! The document names `model.language_model.layers.0.mlp.experts` once and the
//! shard set carries 256 experts times three projections times three tensors
//! under it. So a lookup is a walk from the longest prefix of a tensor name to
//! the shortest, and the first target that matches wins. That is
//! [`QuantConfig::algo_for`], and it is the reason this file holds a map rather
//! than the document's own array order: at 94,393 tensors a linear scan of 291
//! targets per tensor is 27 million string comparisons, and a prefix walk is
//! bounded by the number of dots in a name.
//!
//! # `quant_algo` at the top is not a fact about any tensor
//!
//! It reads `MIXED_PRECISION`, which is the document saying "see
//! `quantized_layers`". It is parsed and kept so a checkpoint that is uniformly
//! one algorithm can still be read, and it is never consulted when
//! `quantized_layers` has an answer.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use shiso_core::error::{CoreError, CoreResult};

/// How one target's weights - and, for FP8, its activations - are stored.
///
/// Three variants and not one enum over every name modelopt can emit: an
/// algorithm this build has no decoder for has to be a refusal naming the
/// spelling, because the alternative is a tensor decoded by the wrong codec
/// into finite numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantAlgo {
    /// FP8 E4M3 weights with FP8 E4M3 activations, one `f32` scale per tensor
    /// on each side.
    ///
    /// This is the first activation quantization in this engine: W8A8, not W8.
    Fp8,
    /// NVFP4 weights at 16-bit activations: E2M1 values, one E4M3 scale per
    /// group of `group_size` values, and one `f32` per tensor above both.
    ///
    /// The third level is the one that is easy to drop and costs a factor of
    /// about 31,458 when it is dropped, which the next normalization very
    /// nearly hides.
    W4A16Nvfp4 {
        /// Values one E4M3 scale covers.
        group_size: u32,
    },
    /// Not quantized at all: the tensor is stored at its declared element type.
    ///
    /// What a target absent from `quantized_layers` means, and what every norm,
    /// bias and embedding table in this checkpoint is.
    None,
}

/// The whole quantization document, read once.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct QuantConfig {
    /// One entry per target the document names, keyed by the target prefix.
    targets: HashMap<String, QuantAlgo>,
    /// What the key/value cache was calibrated for, when the document says.
    kv_cache: Option<QuantAlgo>,
    /// Prefixes the export deliberately left unquantized and unused, as they
    /// are written: a trailing `*` is the only wildcard this document uses.
    exclude: Vec<String>,
}

/// The document's `quantization` block, before it is checked.
#[derive(Debug, Deserialize)]
struct QuantizationRaw {
    #[serde(default)]
    quant_algo: Option<String>,
    #[serde(default)]
    kv_cache_quant_algo: Option<String>,
    #[serde(default)]
    quantized_layers: HashMap<String, TargetRaw>,
    #[serde(default)]
    exclude_modules: Vec<String>,
}

/// One target's entry.
#[derive(Debug, Deserialize)]
struct TargetRaw {
    quant_algo: String,
    #[serde(default)]
    group_size: Option<u32>,
}

/// The whole document.
#[derive(Debug, Deserialize)]
struct DocumentRaw {
    quantization: QuantizationRaw,
}

impl QuantConfig {
    /// The three parts, given directly, for a test that needs a small one.
    #[must_use]
    pub fn new(
        targets: HashMap<String, QuantAlgo>,
        kv_cache: Option<QuantAlgo>,
        exclude: Vec<String>,
    ) -> Self {
        Self {
            targets,
            kv_cache,
            exclude,
        }
    }

    /// Reads the document at `path`.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] when the file is not there,
    /// [`CoreError::Json`] when it does not parse, and
    /// [`CoreError::InvalidValue`] naming the spelling when a target claims an
    /// algorithm this build has no decoder for.
    pub fn from_file(path: &Path) -> CoreResult<Self> {
        let text = std::fs::read_to_string(path).map_err(|_| CoreError::ConfigMissing {
            key: path.display().to_string(),
        })?;
        Self::from_json_str(&text)
    }

    /// Reads the document out of text.
    ///
    /// # Errors
    ///
    /// Whatever [`QuantConfig::from_file`] reports, less the read.
    pub fn from_json_str(text: &str) -> CoreResult<Self> {
        let doc: DocumentRaw = serde_json::from_str(text)?;
        let raw = doc.quantization;
        // The top-level algorithm is checked but not stored as a target: a
        // document that says `MIXED_PRECISION` is deferring to the map below,
        // and one that names an algorithm with no decoder should refuse here
        // rather than at the tensor that first needed it.
        if let Some(algo) = raw.quant_algo.as_deref()
            && algo != "MIXED_PRECISION"
        {
            parse_algo(algo, None)?;
        }
        let kv_cache = raw
            .kv_cache_quant_algo
            .as_deref()
            .map(|algo| parse_algo(algo, None))
            .transpose()?;
        let targets = raw
            .quantized_layers
            .into_iter()
            .map(|(name, target)| Ok((name, parse_algo(&target.quant_algo, target.group_size)?)))
            .collect::<CoreResult<HashMap<String, QuantAlgo>>>()?;
        Ok(Self {
            targets,
            kv_cache,
            exclude: raw.exclude_modules,
        })
    }

    /// How the tensor called `name` is stored.
    ///
    /// The walk is from the longest prefix of `name` to the shortest, cut at
    /// the dots, and the first target that matches wins. A name no target
    /// covers is [`QuantAlgo::None`], which is the right answer for every norm
    /// and bias in the file and is what an absent entry means.
    #[must_use]
    pub fn algo_for(&self, name: &str) -> QuantAlgo {
        let mut cut = name.len();
        loop {
            if let Some(algo) = self.targets.get(&name[..cut]) {
                return *algo;
            }
            match name[..cut].rfind('.') {
                Some(dot) => cut = dot,
                None => return QuantAlgo::None,
            }
        }
    }

    /// Whether the export says this tensor is not part of the model.
    ///
    /// `mtp*` is the draft head: 785 BF16 tensors the checkpoint's own document
    /// excludes, and which this build does not run.
    #[must_use]
    pub fn is_excluded(&self, name: &str) -> bool {
        self.exclude
            .iter()
            .any(|pattern| match pattern.strip_suffix('*') {
                Some(prefix) => name.starts_with(prefix),
                None => name == pattern,
            })
    }

    /// What the key/value cache was calibrated for, when the document says.
    #[must_use]
    pub const fn kv_cache(&self) -> Option<QuantAlgo> {
        self.kv_cache
    }

    /// Targets the document names.
    #[must_use]
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Whether the document names no target at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Targets stored as `algo`, which is how a census is asserted.
    #[must_use]
    pub fn count_of(&self, algo: QuantAlgo) -> usize {
        self.targets.values().filter(|kept| **kept == algo).count()
    }
}

/// Maps a `quant_algo` spelling onto the codec that reads it.
///
/// A spelling with no decoder is [`CoreError::InvalidValue`] carrying the
/// spelling, because that is the only thing a caller can act on.
fn parse_algo(algo: &str, group_size: Option<u32>) -> CoreResult<QuantAlgo> {
    match algo {
        "FP8" => Ok(QuantAlgo::Fp8),
        // `NVFP4` is the same weights with an activation scale calibrated for
        // a four-bit activation pass. No kernel here quantizes activations to
        // four bits, so the target runs at 16-bit activations - closer to the
        // unquantized model than the pass the scale was calibrated for, and
        // the same codec for the weights.
        "W4A16_NVFP4" | "NVFP4" => Ok(QuantAlgo::W4A16Nvfp4 {
            // The document states it on every NVFP4 target this checkpoint
            // carries. A target that omitted it would be a group size guessed
            // at by the decoder, which is a scale read from the wrong offset.
            group_size: group_size.ok_or(CoreError::ConfigMissing {
                key: "quantized_layers.*.group_size".to_string(),
            })?,
        }),
        other => Err(CoreError::InvalidValue {
            key: "quant_algo",
            value: other.to_string(),
        }),
    }
}

#[cfg(test)]
#[path = "quant_tests.rs"]
mod tests;
