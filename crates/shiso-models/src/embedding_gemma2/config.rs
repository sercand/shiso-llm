//! What the container says the encoder is, checked against what this family
//! can run.
//!
//! Read from the GGUF's own `gemma-embedding2.*` metadata rather than a
//! `config.json`: the quantized container is what a deployment holds, and it
//! carries every dimension the encoder needs. The checks refuse a file whose
//! metadata describes something this reader would run wrongly - a causal
//! stack, a pooling other than the mean, a partial rotation - rather than
//! running it and returning vectors that compare fluently and rank nothing.

use shiso_core::TokenId;
use shiso_modelfile::{GgufFile, GgufMetadata, MetaValue};

use crate::embedder::EmbedError;

/// The architecture string the container declares.
pub const GGUF_ARCH: &str = "gemma-embedding2";

/// llama.cpp's `LLAMA_POOLING_TYPE_MEAN`, the only pooling this family was
/// trained with.
const POOLING_MEAN: u32 = 1;

/// One attention layer's geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerDims {
    /// Key and value heads; the query heads are shared across the stack.
    pub kv_heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// Rotary base.
    pub theta: f32,
    /// Keys visible on each side of a query, `None` for the whole sequence.
    pub radius: Option<usize>,
}

/// The token ids that lay a sequence out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Markers {
    /// Opens every sequence.
    pub bos: TokenId,
    /// Closes every sequence.
    pub eos: TokenId,
    /// Opens a picture's or a video frame's soft-token run.
    pub image_open: TokenId,
    /// Closes it.
    pub image_close: TokenId,
    /// One picture soft token.
    pub image: TokenId,
    /// One video-frame soft token.
    pub video: TokenId,
    /// Opens an audio clip's soft-token run.
    pub audio_open: TokenId,
    /// Closes it.
    pub audio_close: TokenId,
    /// One audio soft token.
    pub audio: TokenId,
}

/// The encoder's geometry, as the container declares it.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    /// Residual width.
    pub hidden: usize,
    /// Feed-forward width.
    pub intermediate: usize,
    /// Query heads.
    pub q_heads: usize,
    /// Per-layer geometry, one entry per layer.
    pub layers: Vec<LayerDims>,
    /// Width of each layer's slice of the per-layer signal.
    pub ple_dim: usize,
    /// Width of the pooled output.
    pub out_dim: usize,
    /// Rows of the token table.
    pub vocab: usize,
    /// Every norm's epsilon.
    pub eps: f32,
    /// The sequence layout's ids.
    pub markers: Markers,
}

impl EncoderConfig {
    /// Reads and checks the container's metadata.
    ///
    /// # Errors
    ///
    /// [`EmbedError::Checkpoint`] naming the key that is absent, malformed or
    /// describes something this family cannot run.
    #[expect(
        clippy::too_many_lines,
        reason = "every key's check sits beside the read that needs it"
    )]
    pub fn from_gguf(file: &GgufFile) -> Result<Self, EmbedError> {
        let meta = file.metadata();
        if meta.architecture() != Some(GGUF_ARCH) {
            return Err(bad(format!(
                "general.architecture is {:?}, not {GGUF_ARCH}",
                meta.architecture()
            )));
        }
        // The two facts the rest of this reader assumes, checked before any
        // dimension is read: a file that declares itself causal, or pooled
        // any way but the mean, is a different model behind the same names.
        if meta.get_bool(&key("attention.causal")) != Some(false) {
            return Err(bad(format!(
                "{} is not false; this family's encoder is bidirectional",
                key("attention.causal")
            )));
        }
        let pooling = arch_u32(meta, "pooling_type")?;
        if pooling != POOLING_MEAN {
            return Err(bad(format!(
                "{} is {pooling}; this family pools by the mean ({POOLING_MEAN})",
                key("pooling_type")
            )));
        }
        let n_layers = arch_usize(meta, "block_count")?;
        let hidden = arch_usize(meta, "embedding_length")?;
        let q_heads = arch_usize(meta, "attention.head_count")?;
        let kv_heads = usize_array(meta, "attention.head_count_kv", n_layers)?;
        let sliding = bool_array(meta, "attention.sliding_window_pattern", n_layers)?;
        let global_head = arch_usize(meta, "attention.key_length")?;
        let local_head = arch_usize(meta, "attention.key_length_swa")?;
        // A value width that differs from the key width, or a rotation that
        // covers part of a head, would each need a code path this encoder does
        // not have. The checkpoint states neither; a file that does is
        // refused here rather than run through the wrong one.
        for (value, rotated, width) in [
            (
                "attention.value_length",
                "rope.dimension_count",
                global_head,
            ),
            (
                "attention.value_length_swa",
                "rope.dimension_count_swa",
                local_head,
            ),
        ] {
            for k in [value, rotated] {
                let got = arch_usize(meta, k)?;
                if got != width {
                    return Err(bad(format!(
                        "{} is {got}, not the head width {width}",
                        key(k)
                    )));
                }
            }
        }
        // The container states the window's full width; the reference reads
        // it as a symmetric radius of half that, `|q - k| <= width / 2`.
        let window = arch_usize(meta, "attention.sliding_window")?;
        if window == 0 || !window.is_multiple_of(2) {
            return Err(bad(format!(
                "{} is {window}, not an even width",
                key("attention.sliding_window")
            )));
        }
        if arch_u32(meta, "attention.shared_kv_layers")? != 0 {
            return Err(bad(format!(
                "{} is not 0; this family's encoder shares no key-value layers",
                key("attention.shared_kv_layers")
            )));
        }
        let global_theta = arch_f32(meta, "rope.freq_base")?;
        let local_theta = arch_f32(meta, "rope.freq_base_swa")?;
        let layers = sliding
            .iter()
            .zip(&kv_heads)
            .map(|(&sliding, &kv_heads)| {
                if sliding {
                    LayerDims {
                        kv_heads,
                        head_dim: local_head,
                        theta: local_theta,
                        radius: Some(window / 2),
                    }
                } else {
                    LayerDims {
                        kv_heads,
                        head_dim: global_head,
                        theta: global_theta,
                        radius: None,
                    }
                }
            })
            .collect::<Vec<_>>();
        for (i, layer) in layers.iter().enumerate() {
            if layer.kv_heads == 0 || !q_heads.is_multiple_of(layer.kv_heads) {
                return Err(bad(format!(
                    "layer {i} has {} key-value heads, which do not divide {q_heads} query heads",
                    layer.kv_heads
                )));
            }
        }
        let tokens = meta
            .get_array("tokenizer.ggml.tokens")
            .ok_or_else(|| bad("tokenizer.ggml.tokens is absent".to_owned()))?;
        let vocab = tokens.len();
        let markers = markers(meta, tokens)?;
        Ok(Self {
            hidden,
            intermediate: arch_usize(meta, "feed_forward_length")?,
            q_heads,
            layers,
            ple_dim: arch_usize(meta, "embedding_length_per_layer_input")?,
            out_dim: arch_usize(meta, "embedding_length_out")?,
            vocab,
            eps: arch_f32(meta, "attention.layer_norm_rms_epsilon")?,
            markers,
        })
    }

    /// Layers in the stack.
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

/// The marker ids, found by their spelling in the container's own vocabulary
/// rather than written down: a number copied from one revision's config is the
/// kind of fact that drifts, and a vocabulary that spells a marker differently
/// is refused here instead of embedding around a token the model never saw.
fn markers(
    meta: &GgufMetadata,
    tokens: &shiso_modelfile::MetaArray,
) -> Result<Markers, EmbedError> {
    const SPELLINGS: [&str; 7] = [
        "<|image>",
        "<image|>",
        "<|image|>",
        "<|video|>",
        "<|audio>",
        "<audio|>",
        "<|audio|>",
    ];
    let mut found = [None; SPELLINGS.len()];
    for (index, entry) in tokens.iter().enumerate() {
        let Ok(MetaValue::Str(piece)) = entry else {
            continue;
        };
        if let Some(slot) = SPELLINGS.iter().position(|s| **s == *piece) {
            found[slot] = Some(TokenId::new(u32::try_from(index).unwrap_or(u32::MAX)));
        }
    }
    let id = |slot: usize| {
        found[slot].ok_or_else(|| {
            bad(format!(
                "the vocabulary has no {} token, which this family's layout needs",
                SPELLINGS[slot]
            ))
        })
    };
    let special = |name: &str| {
        meta.get_u32(name)
            .map(TokenId::new)
            .ok_or_else(|| bad(format!("{name} is absent")))
    };
    Ok(Markers {
        bos: special("tokenizer.ggml.bos_token_id")?,
        eos: special("tokenizer.ggml.eos_token_id")?,
        image_open: id(0)?,
        image_close: id(1)?,
        image: id(2)?,
        video: id(3)?,
        audio_open: id(4)?,
        audio_close: id(5)?,
        audio: id(6)?,
    })
}

fn bad(what: String) -> EmbedError {
    EmbedError::Checkpoint { what }
}

fn key(suffix: &str) -> String {
    format!("{GGUF_ARCH}.{suffix}")
}

fn arch_u32(meta: &GgufMetadata, suffix: &str) -> Result<u32, EmbedError> {
    meta.arch_u32(suffix)
        .ok_or_else(|| bad(format!("{} is absent or not an integer", key(suffix))))
}

fn arch_usize(meta: &GgufMetadata, suffix: &str) -> Result<usize, EmbedError> {
    let value = arch_u32(meta, suffix)?;
    if value == 0 {
        return Err(bad(format!("{} is 0", key(suffix))));
    }
    usize::try_from(value).map_err(|_| bad(format!("{} does not fit", key(suffix))))
}

fn arch_f32(meta: &GgufMetadata, suffix: &str) -> Result<f32, EmbedError> {
    meta.arch_f32(suffix)
        .filter(|v| v.is_finite() && *v > 0.0)
        .ok_or_else(|| {
            bad(format!(
                "{} is absent or not a positive number",
                key(suffix)
            ))
        })
}

/// A per-layer array of `n` entries.
fn per_layer<T>(
    meta: &GgufMetadata,
    suffix: &str,
    n: usize,
    read: impl Fn(MetaValue) -> Option<T>,
) -> Result<Vec<T>, EmbedError> {
    let name = key(suffix);
    let array = meta
        .get_array(&name)
        .ok_or_else(|| bad(format!("{name} is absent or not an array")))?;
    if array.len() != n {
        return Err(bad(format!(
            "{name} has {} entries for {n} layers",
            array.len()
        )));
    }
    array
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            entry
                .ok()
                .and_then(&read)
                .ok_or_else(|| bad(format!("{name} entry {i} is malformed")))
        })
        .collect()
}

fn usize_array(meta: &GgufMetadata, suffix: &str, n: usize) -> Result<Vec<usize>, EmbedError> {
    per_layer(meta, suffix, n, |v| match v {
        MetaValue::I32(v) => usize::try_from(v).ok(),
        MetaValue::U32(v) => usize::try_from(v).ok(),
        _ => None,
    })
}

fn bool_array(meta: &GgufMetadata, suffix: &str, n: usize) -> Result<Vec<bool>, EmbedError> {
    per_layer(meta, suffix, n, |v| match v {
        MetaValue::Bool(b) => Some(b),
        _ => None,
    })
}
