//! The documents this family is described by, turned into one geometry.
//!
//! The container is the source of truth: every width the stack runs is in its
//! `clef.*` metadata, and the published `config.json` is read only to recognise
//! the family and to cross-check the widths it repeats.
//!
//! # Only the Clef container, for now
//!
//! The one container this family is read from is a Clef GGUF whose backbone is
//! a dense llama stack ([`ClefBackbone::Llama`]). A stock `llama` GGUF is
//! refused rather than half-supported: its exporter permutes every query and
//! key row so that the rotation pairs adjacent dimensions, and this family's
//! attention rotates the half-split pairs `(i, i + head_dim / 2)` that the
//! checkpoint itself uses - which the Clef exporter leaves in place. Running a
//! permuted file through the half-split rotation would answer fluently and
//! wrongly.
//!
//! A dense llama slot normalizes neither its queries nor its keys, so every
//! attention slot here is [`AttnLayer::without_qk_norm`] with the whole head
//! rotated, which the engine's dense set runs as rotation without a norm.

use std::path::Path;

use serde::Deserialize;
use shiso_core::config::HfConfig;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::TokenId;
use shiso_engine::shape::{
    Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, NormScheme,
};
use shiso_modelfile::{GgufFile, GgufMetadata};

use crate::clef::{self, ClefBackbone};

/// The `model_type` this family answers to.
pub const ARCH: &str = "llama";

/// The architecture a stock export of this family declares, which is refused.
const STOCK_GGUF_ARCH: &str = "llama";

/// The architecture prefix every geometry key is read under.
const GGUF_ARCH: &str = clef::ARCH;

/// The decision head the container must name; the head is the reason this
/// container exists.
const DECISION_KEY: &str = "clef.decision.type";

/// The `config.json` keys [`LlamaDocument`] is read from.
const DOCUMENT_FIELDS: [&str; 7] = [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "tie_word_embeddings",
    "rope_scaling",
];

/// The subset of `config.json` that is cross-checked against the container.
#[derive(Debug, Deserialize)]
struct LlamaDocument {
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: Option<u32>,
    vocab_size: u32,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
}

/// One checkpoint's geometry, checked.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    shape: HybridShape,
    stop_tokens: Vec<TokenId>,
}

impl LlamaConfig {
    /// Whether `config` names this family.
    #[must_use]
    pub fn describes(config: &HfConfig) -> bool {
        config.model_type() == ARCH
    }

    /// Recognises the document, reads the geometry out of the container at
    /// `checkpoint`, and checks every width the document repeats against it.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] when the document describes another family
    /// (the one failure a caller reads as "not mine"), every refusal
    /// [`Self::from_gguf`] makes, [`CoreError::ShapeMismatch`] when the
    /// document and the container disagree, and [`CoreError::Json`] when the
    /// document does not deserialize.
    pub fn from_hf(config: &HfConfig, checkpoint: &Path) -> CoreResult<Self> {
        if !Self::describes(config) {
            return Err(CoreError::UnknownArch {
                model_type: config.model_type().to_owned(),
            });
        }
        let file = GgufFile::open_shared(checkpoint).map_err(|source| CoreError::Backend {
            backend: "llama",
            op: "read the checkpoint's metadata",
            source: Box::new(source),
        })?;
        let family = Self::from_gguf(&file)?;
        // Only the fields cross-checked below, each read as its own subtree:
        // the rest of the document is the exporter's and is not this
        // family's to interpret.
        let mut fields = serde_json::Map::new();
        for key in DOCUMENT_FIELDS {
            if let Ok(value) = config.section(key) {
                fields.insert(key.to_owned(), value.clone());
            }
        }
        let document: LlamaDocument = serde_json::from_value(serde_json::Value::Object(fields))?;
        family.cross_check(&document)?;
        Ok(family)
    }

    /// Reads the geometry out of the container's own metadata.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] for a container that is not a Clef GGUF over
    /// a dense llama backbone, [`CoreError::Unsupported`] for a stock llama
    /// GGUF or a rotation this family does not run, [`CoreError::ConfigMissing`]
    /// for an absent key and [`CoreError::InvalidValue`] for a value the
    /// kernels were not written for.
    pub fn from_gguf(file: &GgufFile) -> CoreResult<Self> {
        recognise(file)?;
        let meta = file.metadata();

        let n_layers = arch_u32(meta, "block_count")?;
        let hidden = arch_u32(meta, "embedding_length")?;
        let q_heads = arch_u32(meta, "attention.head_count")?;
        let kv_heads = arch_u32(meta, "attention.head_count_kv")?;
        if q_heads == 0 || kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
            return Err(invalid(
                "clef.attention.head_count_kv",
                &format!("{q_heads} query heads over {kv_heads} key/value heads"),
            ));
        }
        // The head width is stated; a file that omits it means the stream over
        // the query heads, as the reference derives it.
        let head_dim = match meta.arch_u32("attention.key_length") {
            Some(width) => width,
            None if hidden.is_multiple_of(q_heads) => hidden / q_heads,
            None => return Err(missing("clef.attention.key_length")),
        };
        let value_dim = meta.arch_u32("attention.value_length").unwrap_or(head_dim);
        if value_dim != head_dim {
            return Err(invalid(
                "clef.attention.value_length",
                &format!("{value_dim}, against a key head {head_dim} wide"),
            ));
        }
        // The whole head rotates: the reference rotates every dimension of
        // it, and a narrower rotated prefix is a variant this family was not
        // checked against.
        let n_rot = meta.arch_u32("rope.dimension_count").unwrap_or(head_dim);
        if n_rot != head_dim {
            return Err(invalid(
                "clef.rope.dimension_count",
                &format!("{n_rot} of a {head_dim}-wide head"),
            ));
        }
        let theta = arch_f32(meta, "rope.freq_base")?;
        let slot =
            AttnLayer::new(q_heads, kv_heads, head_dim, n_rot, n_rot, theta).without_qk_norm();
        let layers =
            usize::try_from(n_layers).map_err(|_| invalid("clef.block_count", "too many"))?;

        let vocab = vocab(meta)?;
        let shape = HybridShape {
            layers: vec![LayerKind::Attn; layers],
            attn: vec![slot; layers],
            hidden,
            vocab,
            intermediate: arch_u32(meta, "feed_forward_length")?,
            // No gated delta-rule slot. The convolution width is 1 rather
            // than 0 because it is a divisor wherever the engine reads it.
            gdn: GdnShape::new(0, 0, 0, 1),
            attn_output_gate: false,
            rms_eps: arch_f32(meta, "attention.layer_norm_rms_epsilon")?,
            activation: Activation::SwiGlu,
            // An input norm before attention and one before the feed-forward,
            // and nothing after either.
            norms: NormScheme::PreOnly,
            embed_scale: None,
            final_softcap: None,
            layer_scalar: false,
            ple: None,
            media_positions: MediaPositions::Linear,
            v_norm: false,
            moe: None,
            ssm: None,
            short_conv: None,
            no_ffn: Vec::new(),
        };
        shape.validate()?;

        let eos = meta
            .get_u32("tokenizer.ggml.eos_token_id")
            .ok_or_else(|| missing("tokenizer.ggml.eos_token_id"))?;
        let mut stop_tokens = vec![TokenId::new(eos)];
        if let Some(eot) = meta.get_u32("tokenizer.ggml.eot_token_id")
            && eot != eos
        {
            stop_tokens.push(TokenId::new(eot));
        }
        Ok(Self { shape, stop_tokens })
    }

    /// Refuses a document whose repeated widths disagree with the container.
    fn cross_check(&self, doc: &LlamaDocument) -> CoreResult<()> {
        let shape = &self.shape;
        HfConfig::expect(
            "hidden_size",
            u64::from(doc.hidden_size),
            u64::from(shape.hidden),
        )?;
        HfConfig::expect(
            "num_hidden_layers",
            u64::from(doc.num_hidden_layers),
            u64::from(shape.n_layers()),
        )?;
        HfConfig::expect(
            "vocab_size",
            u64::from(doc.vocab_size),
            u64::from(shape.vocab),
        )?;
        if let Some(slot) = shape.attn.first() {
            HfConfig::expect(
                "num_attention_heads",
                u64::from(doc.num_attention_heads),
                u64::from(slot.q_heads),
            )?;
            // The document's default is one key/value head per query head.
            HfConfig::expect(
                "num_key_value_heads",
                u64::from(doc.num_key_value_heads.unwrap_or(doc.num_attention_heads)),
                u64::from(slot.kv_heads),
            )?;
        }
        if doc.tie_word_embeddings == Some(false) {
            return Err(invalid(
                "tie_word_embeddings",
                "false, and this family's loader ties the logits to the embedding",
            ));
        }
        if doc.rope_scaling.as_ref().is_some_and(|v| !v.is_null()) {
            return Err(CoreError::Unsupported {
                what: "a scaled rotation in the document",
                backend: "llama",
            });
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
}

/// Refuses every container but a Clef GGUF over a dense llama backbone that
/// names the head this build scores with and rotates at its base ladder.
fn recognise(file: &GgufFile) -> CoreResult<()> {
    let meta = file.metadata();
    match meta.architecture() {
        Some(GGUF_ARCH) => {}
        Some(STOCK_GGUF_ARCH) => {
            return Err(CoreError::Unsupported {
                what: "a stock llama GGUF, whose query and key rows are permuted for an \
                   adjacent-pair rotation this family does not run",
                backend: "llama",
            });
        }
        other => {
            return Err(CoreError::UnknownArch {
                model_type: other.unwrap_or_default().to_owned(),
            });
        }
    }
    if ClefBackbone::of(file)? != ClefBackbone::Llama {
        return Err(CoreError::UnknownArch {
            model_type: "clef over the qwen35 hybrid".to_owned(),
        });
    }
    match meta.get_str(DECISION_KEY) {
        Some("clef") => {}
        None => return Err(missing(DECISION_KEY)),
        // A head this build has never seen is refused rather than scored
        // with the one it has.
        Some(other) => return Err(invalid(DECISION_KEY, other)),
    }
    refuse_rope_scaling(meta)?;
    Ok(())
}

/// The vocabulary width: stated, or the count of the token table.
fn vocab(meta: &GgufMetadata) -> CoreResult<u32> {
    if let Some(vocab) = meta.arch_u32("vocab_size") {
        return Ok(vocab);
    }
    let tokens = meta
        .get_array("tokenizer.ggml.tokens")
        .ok_or_else(|| missing("tokenizer.ggml.tokens"))?;
    u32::try_from(tokens.len()).map_err(|_| invalid("tokenizer.ggml.tokens", "too many"))
}

/// Refuses a container that scales its rotation: the kernels rotate at the
/// base frequency ladder and nothing else.
fn refuse_rope_scaling(meta: &GgufMetadata) -> CoreResult<()> {
    let typed = meta
        .arch_str("rope.scaling.type")
        .is_some_and(|kind| kind != "none");
    let factor = meta
        .arch_f32("rope.scaling.factor")
        .is_some_and(|f| (f - 1.0).abs() > f32::EPSILON);
    if typed || factor {
        return Err(CoreError::Unsupported {
            what: "a scaled rotation in the container",
            backend: "llama",
        });
    }
    Ok(())
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
