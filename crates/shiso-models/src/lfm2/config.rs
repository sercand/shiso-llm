//! The documents this family is described by, turned into one geometry.
//!
//! The container is the source of truth: a published `config.json` for a
//! vision-language checkpoint of this family nests the text geometry under
//! `text_config` beside a vision tower the GGUF does not carry, and the GGUF
//! states every width the stack runs. So the document is read only to
//! recognise the family and to cross-check the widths it repeats, and the
//! geometry comes out of the container's `lfm2.*` metadata.
//!
//! # Which slots convolve
//!
//! The container has no layer-type list. It writes a key/value head count per
//! slot instead, and a slot with zero key/value heads is a short-convolution
//! slot: it has no attention to give heads to. The document's `layer_types`
//! says the same thing in words, and when it is present the two are checked
//! against each other, because a slot read as the wrong kind loads tensors of
//! the wrong names - or, worse, of the right names at the wrong slot - and the
//! loader would then refuse a file that is fine or run one that is not.

use std::path::Path;

use serde::Deserialize;
use shiso_core::config::HfConfig;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::TokenId;
use shiso_engine::shape::{
    Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
    ShortConvShape,
};
use shiso_modelfile::{GgufFile, GgufMetadata, MetaValue};

/// The `model_type` spellings this family answers to: the vision-language
/// wrapper and the text-only document.
pub const ARCHES: [&str; 2] = ["lfm2_vl", "lfm2"];

/// The GGUF architecture string.
pub const GGUF_ARCH: &str = "lfm2";

/// The `model_type` of the text subtree, when the document has one.
const TEXT_ARCH: &str = "lfm2";

/// The `layer_types` spelling of a short-convolution slot.
const CONV: &str = "conv";

/// The `layer_types` spelling of an attention slot.
const FULL: &str = "full_attention";

/// The metadata key a decision checkpoint of this family names its readout in.
const DECISION_KEY: &str = "lfm2.decision.type";

/// How a checkpoint of this family is meant to be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lfm2Readout {
    /// A language model: its logits are sampled from.
    Generative,
    /// A decision model that answers typed questions from the next-token
    /// distribution at the answer slot of its own prompt layout, generating
    /// nothing.
    Decision,
}

/// The subset of the text config that is cross-checked against the container.
#[derive(Debug, Deserialize)]
struct Lfm2TextConfig {
    model_type: String,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    vocab_size: u32,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
}

/// One checkpoint's geometry, checked.
#[derive(Debug, Clone)]
pub struct Lfm2Config {
    shape: HybridShape,
    stop_tokens: Vec<TokenId>,
    readout: Lfm2Readout,
}

impl Lfm2Config {
    /// Whether `config` names this family.
    #[must_use]
    pub fn describes(config: &HfConfig) -> bool {
        ARCHES.contains(&config.model_type())
    }

    /// Recognises the document, reads the geometry out of the container at
    /// `checkpoint`, and checks every width the document repeats against it.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] when the document describes another family
    /// (the one failure a caller reads as "not mine"), and every refusal
    /// [`Self::from_gguf`] makes; [`CoreError::ShapeMismatch`] when the
    /// document and the container disagree; [`CoreError::Json`] when the text
    /// subtree does not deserialize.
    pub fn from_hf(config: &HfConfig, checkpoint: &Path) -> CoreResult<Self> {
        if !Self::describes(config) {
            return Err(CoreError::UnknownArch {
                model_type: config.model_type().to_owned(),
            });
        }
        let family = Self::from_gguf(&open(checkpoint)?)?;
        // The wrapper nests the text geometry and is cross-checked. A text-only
        // document is not: the container carries every width either way, and
        // the document is only handed over by subtree.
        let Ok(subtree) = config.section("text_config") else {
            return Ok(family);
        };
        let text: Lfm2TextConfig = serde_json::from_value(subtree.clone())?;
        if text.model_type != TEXT_ARCH {
            return Err(CoreError::UnknownArch {
                model_type: text.model_type,
            });
        }
        family.cross_check(&text)?;
        Ok(family)
    }

    /// Reads the geometry out of the container's own metadata.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] for another architecture,
    /// [`CoreError::ConfigMissing`] for an absent key, and
    /// [`CoreError::InvalidValue`] / [`CoreError::ShapeMismatch`] for a width
    /// the kernels were not written for.
    pub fn from_gguf(file: &GgufFile) -> CoreResult<Self> {
        let meta = file.metadata();
        if meta.architecture() != Some(GGUF_ARCH) {
            return Err(CoreError::UnknownArch {
                model_type: meta.architecture().unwrap_or_default().to_owned(),
            });
        }
        let n_layers = arch_u32(meta, "block_count")?;
        let hidden = arch_u32(meta, "embedding_length")?;
        let q_heads = arch_u32(meta, "attention.head_count")?;
        let kv_per_slot = kv_heads_per_slot(meta, n_layers)?;
        if q_heads == 0 || !hidden.is_multiple_of(q_heads) {
            return Err(CoreError::InvalidValue {
                key: "lfm2.attention.head_count",
                value: format!("{q_heads} heads over a {hidden}-wide stream"),
            });
        }
        // The head width is not stored; the reference derives it as the
        // stream over the query heads, and so does this.
        let head_dim = hidden / q_heads;
        let theta = arch_f32(meta, "rope.freq_base")?;
        let mut layers = Vec::with_capacity(kv_per_slot.len());
        let mut attn = Vec::new();
        for kv in &kv_per_slot {
            if *kv == 0 {
                layers.push(LayerKind::ShortConv);
            } else {
                layers.push(LayerKind::Attn);
                // The whole head rotates, in the half-split pairing, at the
                // default unit scale: `rope(q, pos, theta, head_dim // 2)`
                // rotates every pair.
                attn.push(AttnLayer::new(
                    q_heads, *kv, head_dim, head_dim, head_dim, theta,
                ));
            }
        }
        let tokens = meta
            .get_array("tokenizer.ggml.tokens")
            .ok_or_else(|| missing("tokenizer.ggml.tokens"))?;
        let vocab = arch_u32(meta, "vocab_size")
            .unwrap_or_else(|_| u32::try_from(tokens.len()).unwrap_or(u32::MAX));
        let shape = HybridShape {
            layers,
            attn,
            hidden,
            vocab,
            intermediate: arch_u32(meta, "feed_forward_length")?,
            // No gated delta-rule slot. The convolution width is 1 rather
            // than 0 because it is a divisor wherever the engine reads it.
            gdn: GdnShape::new(0, 0, 0, 1),
            attn_output_gate: false,
            rms_eps: arch_f32(meta, "attention.layer_norm_rms_epsilon")?,
            activation: Activation::SwiGlu,
            norms: NormScheme::PreOnly,
            embed_scale: None,
            final_softcap: None,
            layer_scalar: false,
            ple: None,
            media_positions: MediaPositions::Linear,
            v_norm: false,
            moe: None,
            ssm: None,
            short_conv: Some(ShortConvShape {
                taps: arch_u32(meta, "shortconv.l_cache")?,
            }),
            no_ffn: Vec::new(),
        };
        shape.validate()?;
        let eos = meta
            .get_u32("tokenizer.ggml.eos_token_id")
            .ok_or_else(|| missing("tokenizer.ggml.eos_token_id"))?;
        let readout = match meta.get_str(DECISION_KEY) {
            None => Lfm2Readout::Generative,
            Some("lfm2-d1") => Lfm2Readout::Decision,
            // A readout this build has never seen is refused rather than run
            // as a language model: its logits would be sampled from as if
            // they were text, which is not what the checkpoint was made for.
            Some(other) => {
                return Err(CoreError::InvalidValue {
                    key: DECISION_KEY,
                    value: other.to_owned(),
                });
            }
        };
        Ok(Self {
            shape,
            stop_tokens: vec![TokenId::new(eos)],
            readout,
        })
    }

    /// Refuses a document whose repeated widths disagree with the container.
    fn cross_check(&self, text: &Lfm2TextConfig) -> CoreResult<()> {
        let shape = &self.shape;
        HfConfig::expect(
            "text_config.hidden_size",
            u64::from(text.hidden_size),
            u64::from(shape.hidden),
        )?;
        HfConfig::expect(
            "text_config.num_hidden_layers",
            u64::from(text.num_hidden_layers),
            u64::from(shape.n_layers()),
        )?;
        HfConfig::expect(
            "text_config.vocab_size",
            u64::from(text.vocab_size),
            u64::from(shape.vocab),
        )?;
        if let Some(first) = shape.attn.first() {
            HfConfig::expect(
                "text_config.num_attention_heads",
                u64::from(text.num_attention_heads),
                u64::from(first.q_heads),
            )?;
            HfConfig::expect(
                "text_config.num_key_value_heads",
                u64::from(text.num_key_value_heads),
                u64::from(first.kv_heads),
            )?;
        }
        if let Some(types) = &text.layer_types {
            let kinds = types
                .iter()
                .map(|kind| match kind.as_str() {
                    CONV => Ok(LayerKind::ShortConv),
                    FULL => Ok(LayerKind::Attn),
                    other => Err(CoreError::InvalidValue {
                        key: "text_config.layer_types",
                        value: other.to_owned(),
                    }),
                })
                .collect::<CoreResult<Vec<_>>>()?;
            if kinds != shape.layers {
                return Err(CoreError::InvalidValue {
                    key: "text_config.layer_types",
                    value: "the document's slot kinds disagree with the container's per-slot \
                            key/value head counts"
                        .to_owned(),
                });
            }
        }
        Ok(())
    }

    /// The geometry the stack runs.
    #[must_use]
    pub const fn shape(&self) -> &HybridShape {
        &self.shape
    }

    /// The ids that end a turn.
    #[must_use]
    pub fn stop_tokens(&self) -> &[TokenId] {
        &self.stop_tokens
    }

    /// How the checkpoint is meant to be read.
    #[must_use]
    pub const fn readout(&self) -> Lfm2Readout {
        self.readout
    }
}

/// The key/value head count of every slot, zero for a convolution slot.
///
/// Stored as an array with one entry per slot; a container that wrote a
/// single number would describe a stack with no convolution slots at all,
/// which is not this family.
fn kv_heads_per_slot(meta: &GgufMetadata, n_layers: u32) -> CoreResult<Vec<u32>> {
    let key = "lfm2.attention.head_count_kv";
    let array = meta.get_array(key).ok_or_else(|| missing(key))?;
    let counts = array
        .iter()
        .map(|value| match value {
            Ok(MetaValue::U32(n)) => Ok(n),
            Ok(MetaValue::I32(n)) => u32::try_from(n).map_err(|_| invalid(key, &n.to_string())),
            Ok(MetaValue::U64(n)) => u32::try_from(n).map_err(|_| invalid(key, &n.to_string())),
            Ok(MetaValue::I64(n)) => u32::try_from(n).map_err(|_| invalid(key, &n.to_string())),
            Ok(other) => Err(invalid(key, &format!("{other:?}"))),
            Err(source) => Err(CoreError::Backend {
                backend: GGUF_ARCH,
                op: "read the per-slot key/value head counts",
                source: Box::new(source),
            }),
        })
        .collect::<CoreResult<Vec<u32>>>()?;
    HfConfig::expect(key, counts.len() as u64, u64::from(n_layers))?;
    Ok(counts)
}

/// Opens the container for its metadata.
fn open(checkpoint: &Path) -> CoreResult<GgufFile> {
    GgufFile::open_shared(checkpoint).map_err(|source| CoreError::Backend {
        backend: GGUF_ARCH,
        op: "read the checkpoint's metadata",
        source: Box::new(source),
    })
}

/// One architecture-prefixed count, or the key that is absent.
fn arch_u32(meta: &GgufMetadata, suffix: &str) -> CoreResult<u32> {
    meta.arch_u32(suffix)
        .ok_or_else(|| missing(&format!("{GGUF_ARCH}.{suffix}")))
}

/// One architecture-prefixed float, or the key that is absent.
fn arch_f32(meta: &GgufMetadata, suffix: &str) -> CoreResult<f32> {
    meta.arch_f32(suffix)
        .ok_or_else(|| missing(&format!("{GGUF_ARCH}.{suffix}")))
}

fn missing(key: &str) -> CoreError {
    CoreError::ConfigMissing {
        key: key.to_owned(),
    }
}

fn invalid(key: &'static str, value: &str) -> CoreError {
    CoreError::InvalidValue {
        key,
        value: value.to_owned(),
    }
}
